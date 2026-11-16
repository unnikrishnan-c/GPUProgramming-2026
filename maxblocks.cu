#include <iostream>
#include <cuda_runtime.h>

__global__ void myKernel() {
    // Kernel code goes here
}

int main() {
    int blockSize = 256; // Number of threads per block
    int numBlocks = 0;   // To store the maximum number of blocks

    // Compute the maximum number of active blocks per multiprocessor
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,    // Output the number of blocks
        (const void*)myKernel, // Kernel function pointer
        blockSize,     // Block size (threads per block)
        0               // No dynamic shared memory in this case
    );

    if (err != cudaSuccess) {
        std::cerr << "Error in cudaOccupancyMaxActiveBlocksPerMultiprocessor: "
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Maximum number of blocks per multiprocessor: " << numBlocks << std::endl;
    return 0;
}

