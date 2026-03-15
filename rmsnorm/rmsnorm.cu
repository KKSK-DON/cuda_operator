// rmsnorm_y.cu — RMSNorm optimization exercises
// CPU ref + block_reduce already provided; your job: implement v0, v1, v2...
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

// -----------------------------------------------
// CPU Reference
// -----------------------------------------------
void rmsnorm_cpu(float* in, float* weight, float* out, int batch, int size,
                 float eps) {
    for (int b = 0; b < batch; b++) {
        float cur_sum = 0.f;
        for (int i = 0; i < size; i++) {
            cur_sum += in[b * size + i] * in[b * size + i];
        }
        float rms = 1 / sqrtf(cur_sum / size + eps);
        for (int i = 0; i < size; i++) {
            out[b * size + i] = in[b * size + i] * weight[i] * rms; 
        }
    }
}

// -----------------------------------------------
// Helper: warp-level sum reduction (provided)
// -----------------------------------------------

__device__ float warp_reduce(float val) {
    #pragma unroll
    for (int stride = 16; stride >= 1; stride /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, stride);
    }
    return val;
}

// -----------------------------------------------
// v0:
// -----------------------------------------------
__global__ void rmsnorm_v0(float* in, float* wei, float* out, int batch,
                            int size, float eps) {
    int tid = threadIdx.x;
    int blk_sz = blockDim.x;
    int blk_id = blockIdx.x;
    float cur_sum = 0.f;
    extern __shared__ float share[];
    
    for (int i = tid; i < size; i += blk_sz) {
        cur_sum += powf(in[blk_id * size + i], 2);
    }

    int warp_size = 32;
    int warp_id = tid / warp_size;
    int lane_id = tid % warp_size;

    float global_sum = warp_reduce(cur_sum);
    if (lane_id == 0) share[warp_id] = global_sum;
    __syncthreads();
    int bar = (blk_sz + warp_size -1) / warp_size;
    if (warp_id == 0) {
        global_sum = tid < bar ? share[tid] : 0.f;
        global_sum = warp_reduce(global_sum);
        if (lane_id == 0) share[0] = global_sum;
    }
    __syncthreads();

    // share[0] 是 float，除以 int size 时 C++ 会自动隐式转换为 float / float，结果正确。
    // 写 static_cast<float>(size) 只是显式风格，意图更清晰，防止将来 sum 被改成 int 时变成整数除法。
    // 整数除法陷阱：int(100) / int(3) = 33（丢失小数），float(100) / int(3) = 33.33（正确）
    float rms_factor = rsqrtf(share[0] / static_cast<float>(size) + eps);
    for (int i = tid; i < size; i += blk_sz) {
        out[blk_id * size + i] = in[blk_id * size + i] * rms_factor * wei[i];
    }
}

// -----------------------------------------------
// v1
// -----------------------------------------------

__global__ void rmsnorm_v1(float* in, float* w, float* out, int batch, int size, float eps) {
  float* in_row = in + blockIdx.x * size;
  float* out_row = out + blockIdx.x * size;
  int tid = threadIdx.x;
  float sum = 0.f;
  int float_size = 4;
  int pack_size = size / float_size;
  int pack_offset = pack_size * float_size;
  extern __shared__ float share[];
  float4* in_pack = reinterpret_cast<float4*>(in_row);
  for (int i = tid; i < pack_size; i += blockDim.x) {
    float4 in_flt4 = in_pack[i];
    sum += in_flt4.x * in_flt4.x;
    sum += in_flt4.y * in_flt4.y;
    sum += in_flt4.z * in_flt4.z;
    sum += in_flt4.w * in_flt4.w;
  }
  for (int i = pack_offset + tid; i < size; i += blockDim.x){
    sum += in_row[i] * in_row[i];
  }
  int warp_size = 32;
  int warp_id = tid / warp_size;
  int lane_id = tid % warp_size;
  float global_sum = warp_reduce(sum);
  if (lane_id == 0) share[warp_id] = global_sum;
  __syncthreads();
  int bar = (blockDim.x + warp_size - 1) / warp_size;
  if (warp_id == 0) {
    global_sum = tid < bar ? share[tid] : 0.f;
    global_sum = warp_reduce(global_sum);
    if (lane_id == 0) share[0] = global_sum;
  }
  __syncthreads();

  float rms = rsqrtf(share[0] / static_cast<float>(size) + eps);
  float4* w_pack = reinterpret_cast<float4*>(w);
  float4* out_pack = reinterpret_cast<float4*>(out_row);
  for (int i = tid; i < pack_size; i += blockDim.x) {
    out_pack[i] = make_float4(in_pack[i].x * rms * w_pack[i].x,
                              in_pack[i].y * rms * w_pack[i].y,
                              in_pack[i].z * rms * w_pack[i].z,
                              in_pack[i].w * rms * w_pack[i].w);
  }
  for (int i = tid + pack_offset; i < size; i += blockDim.x) {
    out_row[i] = in_row[i] * rms * w[i];
  }
}

// -----------------------------------------------
// Correctness check
// -----------------------------------------------
float max_error(const std::vector<float>& ref, const std::vector<float>& got) {
  float err = 0.0f;
  for (int i = 0; i < (int)ref.size(); ++i)
    err = std::max(err, std::abs(ref[i] - got[i]));
  return err;
}

// -----------------------------------------------
// Timing helper — only measures kernel, not memcpy
// Returns average time in microseconds
// -----------------------------------------------
float time_kernel_us(cudaEvent_t ev_start, cudaEvent_t ev_stop, int iters,
                     std::function<void()> launch_fn) {
  // warmup
  for (int i = 0; i < 0; ++i) launch_fn();
  cudaDeviceSynchronize();

  cudaEventRecord(ev_start);
  for (int i = 0; i < iters; ++i) launch_fn();
  cudaEventRecord(ev_stop);
  cudaEventSynchronize(ev_stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, ev_start, ev_stop);
  return ms * 1000.0f / iters;  // microseconds per call
}

// -----------------------------------------------
// Main
// -----------------------------------------------
int main() {
  const int batch = 512;
  const int size = 4096;
  const float eps = 1e-6f;
  const int total = batch * size;
  const int iters = 1;

  // Host data
  std::vector<float> h_in(total), h_wei(size);
  std::vector<float> h_ref(total), h_out(total);

  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.f, 1.f);
  for (auto& v : h_in) v = dist(gen);
  for (auto& v : h_wei) v = dist(gen);

  // CPU reference
  auto cpu_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
    rmsnorm_cpu(h_in.data(), h_wei.data(), h_ref.data(), batch, size, eps);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  float cpu_us = std::chrono::duration<float, std::micro>(cpu_end - cpu_start).count() / iters;
  printf("cpu | time: %7.2f us\n", cpu_us);

  // Device alloc & upload (not timed)
  float *d_in, *d_wei, *d_out;
  cudaMalloc(&d_in, total * sizeof(float));
  cudaMalloc(&d_wei, size * sizeof(float));
  cudaMalloc(&d_out, total * sizeof(float));
  cudaMemcpy(d_in, h_in.data(), total * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wei, h_wei.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 grid(batch), block(1024);

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);

  // ---- v0 ----
  size_t share_size = block.x / 32 * sizeof(float);
  float t0 = time_kernel_us(ev0, ev1, iters, [&]() {
    rmsnorm_v0<<<grid, block, share_size>>>(d_in, d_wei, d_out, batch, size, eps);
  });
  cudaMemcpy(h_out.data(), d_out, total * sizeof(float),
             cudaMemcpyDeviceToHost);
  printf("v0  | time: %7.2f us | max_err: %.2e\n", t0,
         max_error(h_ref, h_out));

  // ---- v1 ----
  float t1 = time_kernel_us(ev0, ev1, iters, [&]() {
    rmsnorm_v1<<<grid, block, share_size>>>(d_in, d_wei, d_out, batch, size, eps);
  });
  cudaMemcpy(h_out.data(), d_out, total * sizeof(float),
             cudaMemcpyDeviceToHost);
  printf("v1  | time: %7.2f us | max_err: %.2e\n", t1,
         max_error(h_ref, h_out));

  // ---- v2 ----
//   float t2 = time_kernel_us(ev0, ev1, iters, [&]() {
//   });
//   cudaMemcpy(h_out.data(), d_out, total * sizeof(float),
//              cudaMemcpyDeviceToHost);
//   printf("v2  | time: %7.2f us | max_err: %.2e\n", t2,
//          max_error(h_ref, h_out));

  cudaFree(d_in);
  cudaFree(d_wei);
  cudaFree(d_out);
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  return 0;
}
