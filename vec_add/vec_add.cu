#include <cstdio>

#define BLOCK_SIZE 256

__global__ void elementwise_add(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 10000000;
    size_t size = N * sizeof(float);
    float *h_a, *h_b, *h_c;

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)i;
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    elementwise_add<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: Expected %d, Got %d\n", i, h_a[i] + h_b[i], h_c[i]);
            break;
        }
    }
    printf("Vector addition completed successfully.\n");

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}


