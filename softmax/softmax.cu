#include <cstdlib>
#include <cstdio>
#include <chrono> 
#include <iostream>      // std::cout
#include <cmath>         // expf, fabsf
#include <cuda_runtime.h> // cudaMalloc, cudaMemcpy, cudaFree, kernel launch

// CPU implementation
void softmax_forward_cpu(float *out, const float *inp, int N, int C) {
  for (int i = 0; i < N; i++) {
    const float *inp_row = inp + i * C;
    float *out_row = out + i * C;

    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
      if (inp_row[j] > maxval) {
        maxval = inp_row[j];
      }
    }
    float sum = 0.f;
    for (int j = 0; j < C; j++) {
      out_row[j] = expf(inp_row[j] - maxval);
      sum += out_row[j];
    }
    float norm = 1.f / sum;
    for (int j = 0; j < C; j++) {
      out_row[j] *= norm;
    }
  }
}

__global__ void softmax_v0(float *in, float *out, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
      return;
    }
    float *input_row = in + idx * C;
    float *output_row = out + idx * C;

    float maxval = -INFINITY;
    for (int i = 0; i < C; i++) {
      if (maxval < input_row[i]) {
        maxval = input_row[i];
      }
    }

    float sum = 0.f;
    for (int i = 0; i < C; i++) {
      output_row[i] = expf(input_row[i] - maxval);
      sum += output_row[i];
    }

    float factor = 1.f / sum;
    for (int i = 0; i < C; i++) {
      output_row[i] *= factor;
    }
}

// reduction
__global__ void softmax_v1(float *in, float *out, int N, int C) {
  int row_idx = blockIdx.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x;
  float *row = in + row_idx * C;
  float *out_row = out + row_idx * C;
  extern __shared__ float shared[];
  // find max in thread level
  float maxval = -INFINITY;
  for (int i = tid; i < C; i += block_size) {
    maxval = fmaxf(maxval, row[i]);
  }
  shared[tid] = maxval;
  __syncthreads();
  // find global max
  for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  float global_max = shared[0];
  for (int i = tid; i < C; i += block_size) {
    out_row[i] = expf(row[i] - global_max);
  }
  float sumval = 0.f;
  for (int i = tid; i < C; i += block_size) {
    sumval += out_row[i];
  }
  shared[tid] = sumval;
  __syncthreads();
  for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  float global_sum = shared[0];
  for (int i = tid; i < C; i += block_size) {
    out_row[i] /= global_sum;
  }
}

__device__ float warp_reduce_max(float val) {
  #pragma unroll
  for (int offset = 16; offset >= 1; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

__device__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = 16; offset >= 1; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

__global__ void softmax_v2(float *in, float *out, int N, int C) {
  int block_id = blockIdx.x;
  int block_size = blockDim.x;
  int tid = threadIdx.x;
  float *x = in + block_id * C;

  float maxval = -INFINITY;
  for (int i = tid; i < C; i += block_size) {
    maxval = fmaxf(maxval, x[i]);
  }
  // __syncthreads(); block level sync, which means sync warp, only one warp so no need

  float global_max = warp_reduce_max(maxval);
  // __shfl_sync one warp all thread broadcast
  global_max = __shfl_sync(0xFFFFFFFF, global_max, 0);
  for (int i = tid; i < C; i += block_size) {
    out[i + block_id * C] = expf(x[i] - global_max);
  }

  float sumval = 0.f;
  for (int i = tid; i < C; i += block_size) {
    sumval += out[i + block_id * C];
  }
  // __syncthreads(); block level sync, which means sync warp, only one warp so no need
  float global_sum = warp_reduce_sum(sumval);
  // __shfl_sync one warp all thread broadcast
  global_sum = __shfl_sync(0xFFFFFFFF, global_sum, 0);

  for (int i = tid; i < C; i += block_size) {
    out[i + block_id * C] /= global_sum;
  }
}

__global__ void softmax_v3(float *in, float *out, int N, int C) {
  // block size 128
  // c 4096
  // one warp handle 32 thread * 32 data = 1024 data
  extern __shared__ float shared[];
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x;
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  float *x = in + bid * C;
  float *y = out + bid * C;

  int warp_size = block_size / 32;
  float *max_arr = shared;
  float *sum_arr = shared + warp_size;
  
  // max
  float maxval = -INFINITY;
  for (int i = tid; i < C; i += block_size) {
    maxval = fmaxf(maxval, x[i]);
  }
  maxval = warp_reduce_max(maxval);
  if (lane_id == 0) {
    max_arr[warp_id] = maxval;
  }
  __syncthreads();
  if (warp_id == 0) {
    for (int stride = warp_size / 2; stride >= 1; stride /= 2) {
      if (tid < stride) {
        shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
      }
    }
  }
  __syncthreads();
  for (int i = tid; i < C; i += block_size) {
    y[i] = expf(x[i] - max_arr[0]);
  }
  //sum
  float sumval = 0.f;
  for (int i = tid; i < C; i += block_size) {
    sumval += y[i];
  }
  sumval = warp_reduce_sum(sumval);
  if (lane_id == 0) {
    sum_arr[warp_id] = sumval;
  }
  __syncthreads();
  if (warp_id == 0) {
    for (int stride = warp_size / 2; stride >= 1; stride /= 2) {
      if (tid < stride) {
        sum_arr[tid] += sum_arr[tid + stride];
      }
    }
  }
  __syncthreads();
  for (int i = tid; i < C; i += block_size) {
    y[i] /= sum_arr[0];
  }
}

__global__ void softmax_v4(float *in, float *out, int N, int C) {
  // block size 128
  // c 4096
  // one warp handle 32 thread * 32 data = 1024 data
  extern __shared__ float shared[];
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x;
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  float *x = in + bid * C;
  float *y = out + bid * C;

  int warp_size = block_size / 32;
  float *max_arr = shared;
  float *sum_arr = shared + warp_size;
  
  // max
  float maxval = -INFINITY;
  for (int i = tid; i < C; i += block_size) {
    maxval = fmaxf(maxval, x[i]);
  }
  maxval = warp_reduce_max(maxval);
  if (lane_id == 0) {
    max_arr[warp_id] = maxval;
  }
  __syncthreads();
  maxval = (tid < warp_size) ? max_arr[tid] : -INFINITY;
  if (warp_id == 0) {
    maxval = warp_reduce_max(maxval);
    if (lane_id == 0) max_arr[0] = maxval;
  }
  __syncthreads();
  float sumval = 0.f;
  for (int i = tid; i < C; i += block_size) {
    y[i] = expf(x[i] - max_arr[0]);
    sumval += y[i];
  }
  //sum
  sumval = warp_reduce_sum(sumval);
  if (lane_id == 0) {
    sum_arr[warp_id] = sumval;
  }
  __syncthreads();
  sumval = (tid < warp_size) ? sum_arr[tid] : 0.f;
  if (warp_id == 0) {
    sumval = warp_reduce_sum(sumval);
    if (lane_id == 0) sum_arr[0] = sumval;
  }
  __syncthreads();
  for (int i = tid; i < C; i += block_size) {
    y[i] /= sum_arr[0];
  }
}

// Function to compare results
bool compare_results(const float *cpu, const float *gpu, int N, int C,
                     float epsilon = 1e-3f) {
  for (int i = 0; i < N * C; ++i) {
    if (fabs(cpu[i] - gpu[i]) > epsilon) {
      std::cout << "Difference at index " << i << ": CPU=" << cpu[i]
                << ", GPU=" << gpu[i] << ", diff=" << fabs(cpu[i] - gpu[i])
                << std::endl;
      return false;
    }
  }
  return true;
}

int main() {
    int N = 1024;
    int C = 4096;
    int BLOCK_SIZE = 256;
    //int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // v0
    int grid_size = N; // v1
    size_t num_of_ele = N * C;

    float *h_in = (float*)malloc(num_of_ele * sizeof(float));
    float *h_out_base = (float*)malloc(num_of_ele * sizeof(float));
    float *h_out = (float*)malloc(num_of_ele * sizeof(float));
    // Initialize input with sample data
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            h_in[n * C + c] = float(c);
        }
    }

    // Run CPU version and measure time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    softmax_forward_cpu(h_out_base, h_in, N, C);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, num_of_ele * sizeof(float));
    cudaMalloc((void**)&d_out, num_of_ele * sizeof(float));
    cudaMemcpy(d_in, h_in, num_of_ele * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    // softmax_v0<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, N, C);
    // softmax_v1<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_in, d_out, N, C);
    // softmax_v2<<<grid_size, 32>>>(d_in, d_out, N, C);
    // softmax_v3<<<grid_size, 128, 128 * sizeof(float) * 2>>>(d_in, d_out, N, C);
    softmax_v4<<<grid_size, 128, 128 * sizeof(float) * 2>>>(d_in, d_out, N, C);
    cudaEventRecord(end);
    // Wait for the event to complete
    cudaEventSynchronize(end);

    cudaMemcpy(h_out, d_out, num_of_ele * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    bool success = compare_results(h_out_base, h_out, N, C);
    std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;
    
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, end);
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time.count() / (gpu_time_ms)) << "x"
            << std::endl;

    free(h_in);
    free(h_out_base);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}