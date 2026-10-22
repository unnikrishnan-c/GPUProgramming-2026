#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h> //header for __nv_bfloat16

__global__ void bf16VecAdd( __nv_bfloat16* a,  __nv_bfloat16* b, __nv_bfloat16* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
        
    }
}

int main() {
    const int N = 4;
    size_t size = N * sizeof(__nv_bfloat16);

    // Host vectors
    std::vector<float> h_a = {2.5f, 4.5f, 5.5f, 6.5f};
    std::vector<float> h_b = {1.5f, 2.5f, 3.5f, 5.0f};
    std::vector<float> h_c(N, 0.0f);

    // Host vectors for  __nv_bfloat16 data
    std::vector<__nv_bfloat16> h_a_bf16(N);
    std::vector<__nv_bfloat16> h_b_bf16(N);
    std::vector<__nv_bfloat16> h_c_bf16(N);

    // Convert host float data to __nv_bfloat16 using __float2bfloat16()
    for (int i = 0; i < N; ++i) {
        h_a_bf16[i] = __float2bfloat16(h_a[i]);
        h_b_bf16[i] = __float2bfloat16(h_b[i]);
    }

    // Allocate Device Memory
    __nv_bfloat16 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from Host to Device
    cudaMemcpy(d_a, h_a_bf16.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b_bf16.data(), size, cudaMemcpyHostToDevice);

    bf16VecAdd<<<1, N>>>(d_a, d_b, d_c, N);

    // Copy result back to Host
    cudaMemcpy(h_c_bf16.data(), d_c, size, cudaMemcpyDeviceToHost);

    // Convert results to float
    for (int i = 0; i < N; ++i) {
        h_c[i] = __bfloat162float(h_c_bf16[i]);
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    // free gpu memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

