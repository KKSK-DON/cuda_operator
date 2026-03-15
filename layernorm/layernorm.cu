// layernorm_y.cu -- LayerNorm learning template
// Compare this with rmsnorm_y.cu and fill the TODOs step by step.
#include <cuda_runtime.h>

#include <algorithm>
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
void layernorm_cpu(float* in, float* weight, float* bias, float* out, int batch,
                   int size, float eps) {
  for (int b = 0; b < batch; b++) {
    float sum = 0.f;
    for (int i = 0; i < size; i++) {
      sum += in[b * size + i];
    }
    float mean = sum / size;
    float var = 0.f;
    for (int i = 0; i < size; i++) {
      var += powf(in[b * size + i] - mean, 2);
    }
    var /= size;
    float factor = rsqrtf(var + eps);
    for (int i = 0; i < size; i++) {
      out[b * size + i] = weight[i] * (in[b * size + i] - mean) * factor + 
                          bias[i];
    }
  }
}

// -----------------------------------------------
// Helper: warp-level sum reduction
// Same idea as rmsnorm_y.cu
// -----------------------------------------------

__device__ float warp_reduce_sum(float val) {
  for (int stride = 16; stride >= 1; stride /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, stride);
  }
  return val;
}

// -----------------------------------------------
// Helper: block-level sum reduction
// share should have at least ceil(blockDim.x / 32) floats
// -----------------------------------------------
__device__ float block_reduce_sum(float val, float* share, int offset) {
  val = warp_reduce_sum(val);
  int warp_size = 32;
  int bsize = blockDim.x;
  int tid = threadIdx.x;
  int warp_id = tid / warp_size;
  int lane = tid % warp_size;
  if (lane == 0) share[offset + warp_id] = val;
  __syncthreads();
  int bar = (bsize + warp_size - 1) / warp_size;
  if (warp_id == 0) {
    val = tid < bar ? share[offset + tid] : 0.f;
    val = warp_reduce_sum(val);
    if (lane == 0) share[offset] = val;
  }
  __syncthreads();
  return share[offset];
}

// -----------------------------------------------
// v0 one block per batch
// -----------------------------------------------
__global__ void layernorm_v0(float* in, float* weight, float* bias, float* out,
                             int batch, int size, float eps) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bsize = blockDim.x;
  float* in_row = in + bid * size;
  float* out_row = out + bid * size;
  extern __shared__ float share[];
  float sum = 0.f;
  for (int i = tid; i < size; i += bsize) {
    sum += in_row[i];
  }
  sum = block_reduce_sum(sum, share, 0);
  float mean = sum / size;
  float var_sum = 0.f;
  for (int i = tid; i < size; i += bsize) {
    var_sum += (in_row[i] - mean) * (in_row[i] - mean);
  }
  var_sum = block_reduce_sum(var_sum, share, 0);
  float factor = rsqrtf(var_sum / static_cast<float>(size) + eps);
  for (int i = tid; i < size; i += bsize) {
    out_row[i] = weight[i] * (in_row[i] - mean) * factor + bias[i];
  }
}

__device__ void block_reduce_for_v1(float sum, float sum_sq, float* share, int offset) {
  sum = warp_reduce_sum(sum);
  sum_sq = warp_reduce_sum(sum_sq);
  int warp_size = 32;
  int bsize = blockDim.x;
  int tid = threadIdx.x;
  int warp_id = tid / warp_size;
  int lane = tid % warp_size;
  int num_warps = (bsize + warp_size - 1) / warp_size;  // = 32
  if (lane == 0) {
    share[offset + warp_id] = sum;                  // offset+0..31
    share[offset + num_warps + warp_id] = sum_sq;   // offset+32..63
  }
  __syncthreads();
  int bar = (bsize + warp_size - 1) / warp_size;
  if (warp_id == 0) {
    sum = tid < bar ? share[offset + tid] : 0.f;
    sum_sq = tid < bar ? share[offset + num_warps + tid] : 0.f;
    sum = warp_reduce_sum(sum);
    sum_sq = warp_reduce_sum(sum_sq);
    if (lane == 0) {
      share[offset] = sum;
      share[offset + 1] = sum_sq;
    }
  }
  __syncthreads();
  return;
}

// -----------------------------------------------
// v1
// -----------------------------------------------
__global__ void layernorm_v1(float* in, float* weight, float* bias, float* out,
                             int batch, int size, float eps) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bsize = blockDim.x;
  float* in_row = in + bid * size;
  float* out_row = out + bid * size;
  extern __shared__ float share[];
  int float4_size = 4;
  int num_float4 = size / float4_size;
  int pack_offset = num_float4 * float4_size;
  float4* in_pack = reinterpret_cast<float4*>(in_row);
  float4* share_pack = reinterpret_cast<float4*>(share);
  float sum = 0.f;
  float sum_sq = 0.f;
  for (int i = tid; i < num_float4; i += bsize) {
    float float_x = in_pack[i].x;
    float float_y = in_pack[i].y;
    float float_z = in_pack[i].z;
    float float_w = in_pack[i].w;
    share_pack[i] = in_pack[i];
    sum += float_x;
    sum += float_y;
    sum += float_z;
    sum += float_w;
    sum_sq += float_x * float_x;
    sum_sq += float_y * float_y;
    sum_sq += float_z * float_z;
    sum_sq += float_w * float_w;
  }
  for (int i = tid + pack_offset; i < size; i += bsize) {
    sum += in_row[i];
    share[i] = in_row[i];
    sum_sq += share[i] * share[i];
  }
  block_reduce_for_v1(sum, sum_sq, share, size);
  float mean;
  float var;
  mean = share[size] / size;
  var = share[size + 1] / size - mean * mean;
  // float mean = sum / size;
  // float var_sum = 0.f;
  // for (int i = tid; i < num_float4; i += bsize) {
  //   var_sum += (share_pack[i].x - mean) * (share_pack[i].x - mean);
  //   var_sum += (share_pack[i].y - mean) * (share_pack[i].y - mean);
  //   var_sum += (share_pack[i].z - mean) * (share_pack[i].z - mean);
  //   var_sum += (share_pack[i].w - mean) * (share_pack[i].w - mean);
  // }
  // for (int i = tid + pack_offset; i < size; i += bsize) {
  //   var_sum += (share[i] - mean) * (share[i] - mean);
  // }
  // var_sum = block_reduce_sum(var_sum, share, size);
  float factor = rsqrtf(var + eps);
  float4* out_pack = reinterpret_cast<float4*>(out_row);
  float4* w_pack = reinterpret_cast<float4*>(weight);
  float4* b_pack = reinterpret_cast<float4*>(bias);
  for (int i = tid; i < num_float4; i += bsize) {
    float float_x = w_pack[i].x * (share_pack[i].x - mean) * factor + b_pack[i].x;
    float float_y = w_pack[i].y * (share_pack[i].y - mean) * factor + b_pack[i].y;
    float float_z = w_pack[i].z * (share_pack[i].z - mean) * factor + b_pack[i].z;
    float float_w = w_pack[i].w * (share_pack[i].w - mean) * factor + b_pack[i].w;
    out_pack[i] = make_float4(float_x, float_y, float_z, float_w);
  }
  for (int i = tid + pack_offset; i < size; i += bsize) {
    out_row[i] = weight[i] * (share[i] - mean) * factor + bias[i];
  }
}

// -----------------------------------------------
// v2: single-pass + float4, no shared memory cache
// -----------------------------------------------
__global__ void layernorm_v2(float* in, float* weight, float* bias, float* out,
                             int batch, int size, float eps) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bsize = blockDim.x;
  float* in_row = in + bid * size;
  float* out_row = out + bid * size;
  extern __shared__ float share[];
  int num_float4 = size / 4;
  float4* in_pack = reinterpret_cast<float4*>(in_row);

  // Pass 1: accumulate sum and sum_sq simultaneously
  float sum = 0.f;
  float sum_sq = 0.f;
  for (int i = tid; i < num_float4; i += bsize) {
    float4 v = in_pack[i];
    sum += v.x + v.y + v.z + v.w;
    sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  // Reduce both values
  block_reduce_for_v1(sum, sum_sq, share, 0);
  float mean = share[0] / size;
  float var = share[1] / size - mean * mean;
  float factor = rsqrtf(var + eps);

  // Pass 2: write output (re-read in_row from global, L1 cache will help)
  float4* out_pack = reinterpret_cast<float4*>(out_row);
  float4* w_pack = reinterpret_cast<float4*>(weight);
  float4* b_pack = reinterpret_cast<float4*>(bias);
  for (int i = tid; i < num_float4; i += bsize) {
    float4 v = in_pack[i];
    float4 w = w_pack[i];
    float4 b = b_pack[i];
    out_pack[i] = make_float4(
        w.x * (v.x - mean) * factor + b.x,
        w.y * (v.y - mean) * factor + b.y,
        w.z * (v.z - mean) * factor + b.z,
        w.w * (v.w - mean) * factor + b.w);
  }
}

// -----------------------------------------------
// Correctness check
// -----------------------------------------------
float max_error(const std::vector<float>& ref, const std::vector<float>& got) {
  float err = 0.0f;
  for (int i = 0; i < static_cast<int>(ref.size()); ++i) {
    err = std::max(err, std::abs(ref[i] - got[i]));
  }
  return err;
}

// -----------------------------------------------
// Timing helper -- only measures kernel, not memcpy
// Returns average time in microseconds
// -----------------------------------------------
float time_kernel_us(cudaEvent_t ev_start, cudaEvent_t ev_stop, int iters,
                     std::function<void()> launch_fn) {
  for (int i = 0; i < 0; ++i) launch_fn();
  cudaDeviceSynchronize();

  cudaEventRecord(ev_start);
  for (int i = 0; i < iters; ++i) launch_fn();
  cudaEventRecord(ev_stop);
  cudaEventSynchronize(ev_stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, ev_start, ev_stop);
  return ms * 1000.0f / iters;
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

  std::vector<float> h_in(total), h_weight(size), h_bias(size);
  std::vector<float> h_ref(total), h_out(total);

  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (auto& v : h_in) v = dist(gen);
  for (auto& v : h_weight) v = dist(gen);
  for (auto& v : h_bias) v = dist(gen);

  auto cpu_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    layernorm_cpu(h_in.data(), h_weight.data(), h_bias.data(), h_ref.data(),
                  batch, size, eps);
  }
  auto cpu_end = std::chrono::high_resolution_clock::now();
  float cpu_us =
      std::chrono::duration<float, std::micro>(cpu_end - cpu_start).count() /
      iters;
  printf("cpu | time: %7.2f us\n", cpu_us);

  float *d_in, *d_weight, *d_bias, *d_out;
  cudaMalloc(&d_in, total * sizeof(float));
  cudaMalloc(&d_weight, size * sizeof(float));
  cudaMalloc(&d_bias, size * sizeof(float));
  cudaMalloc(&d_out, total * sizeof(float));
  cudaMemcpy(d_in, h_in.data(), total * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, h_bias.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 grid(batch), block(1024);
  size_t share_size = ((block.x + 31) / 32) * sizeof(float);

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);

  float t0 = time_kernel_us(ev0, ev1, iters, [&]() {
    layernorm_v0<<<grid, block, share_size>>>(d_in, d_weight, d_bias, d_out,
                                              batch, size, eps);
  });
  cudaMemcpy(h_out.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost);
  printf("v0  | time: %7.2f us | max_err: %.2e\n", t0,
         max_error(h_ref, h_out));
  size_t share_size_v1 = (size + ((block.x + 31) / 32) * 2) * sizeof(float);
  float t1 = time_kernel_us(ev0, ev1, iters, [&]() {
    layernorm_v1<<<grid, block, share_size_v1>>>(d_in, d_weight, d_bias, d_out,
                                              batch, size, eps);
  });
  cudaMemcpy(h_out.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost);
  printf("v1  | time: %7.2f us | max_err: %.2e\n", t1,
         max_error(h_ref, h_out));

  // v2: single-pass (sum + sum_sq), float4, no shared memory cache
  size_t share_size_v2 = ((block.x + 31) / 32) * 2 * sizeof(float);
  float t2 = time_kernel_us(ev0, ev1, iters, [&]() {
    layernorm_v2<<<grid, block, share_size_v2>>>(d_in, d_weight, d_bias, d_out,
                                              batch, size, eps);
  });
  cudaMemcpy(h_out.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost);
  printf("v2  | time: %7.2f us | max_err: %.2e\n", t2,
         max_error(h_ref, h_out));

  cudaFree(d_in);
  cudaFree(d_weight);
  cudaFree(d_bias);
  cudaFree(d_out);
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  return 0;
}
