#include <cmath>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

#define N 4096  // 矩阵大小 N x N
#define M 2048

// CPU reference implementation
void matmul_cpu(float *A, float *B, float *C, int n, int m) {
   for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < m; k++) {
                C[i * n + j] += A[i * m + k]*B[k * n + j];
            }
        }
   }
}

__global__
void matmul_v0(int m, int n, int k, float alpha, float *A, float *B,
                           float beta, float *C){
    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gtidx >= n || gtidy >= n) return;

    float temp = 0.f;
    for (int i = 0; i < k; i++) {
        temp += A[gtidy * k + i] * B[i * n + gtidx];
    }
    C[gtidy * n + gtidx] = alpha * temp + beta * C[gtidy * n + gtidx];
}


bool compare_results(const float *cpu, const float *gpu, int n, float epsilon = 1e-3f) {
    for (int i = 0; i < n * n; ++i) {
        if (fabs(cpu[i] - gpu[i]) > epsilon) {
            std::cout << "Difference at index " << i
                      << ": CPU=" << cpu[i] << ", GPU=" << gpu[i]
                      << ", diff=" << fabs(cpu[i] - gpu[i]) << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    size_t size = N * M * sizeof(float);
    size_t c_size = N * N * sizeof(float);

    float *h_A     = (float *)malloc(size);
    float *h_B     = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(c_size);
    float *h_C_gpu = (float *)malloc(c_size);

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < M; col++) {
            h_A[row * M + col] = 1.0f;
        }
    }
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            h_B[row * N + col] = 2.0f;
        }
    }

    // CPU reference
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C_cpu, N, M);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);

    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, c_size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, c_size);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);
    matmul_v0<<<blocks, threads>>>(N, N, M, 1.f, d_A, d_B, 0.f, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaMemcpy(h_C_gpu, d_C, c_size, cudaMemcpyDeviceToHost);

    bool success = compare_results(h_C_cpu, h_C_gpu, N);
    std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time.count() / gpu_time_ms) << "x" << std::endl;

    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
