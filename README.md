CS5658- GPU Programming 2026
-------------------------------
This repository contains few CUDA programs that is discussed in course lectures and tutorials
Demos of the code is shown in RTX 2080 Ti

1) Why barrier for the grid was working for 1088 thread-blocks when threads-per-block was 32?

2)Whay barrier worked only for 68 thread-blocks when threads-per-block was 1024?

3)How to check runtime errors using cuda-gdb?

4) How to allocate head and head->dist in device for the below code
      struct node {int *dist}*head;
 
5) cuda keywords: __global__ , __device__ , __shared__, __constant__, 

6) cuda datatypes: cudaError_t ,dim3   

7) CUDA API functions: cudaMalloc(),cudaFree(), cudaMemcpy, cudaMemcpyFromSymbol, cudaMemcpyToSymbol,__syncthreads(), atomicAdd, atomicCAS, 

8) builtin variables: threadIdx.x, blockIdx.x, blockDim.x 

