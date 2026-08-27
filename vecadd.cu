#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(int *a, int *b, int *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 4096;
 
    // Host input vectors
    int *h_a;
    int *h_b;
    //Host output vector
    int *h_c;
 
    // Device input vectors
    int *d_a;
    int *d_b;
    //Device output vector
    int *d_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(int);
 
    // Allocate memory for each vector on host
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
 
    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = i;//0..4095
        h_b[i] = 1;
    }
 
    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
 
    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 512;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);
   cudaError_t err;
   err=cudaGetLastError();
    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
   err=cudaGetLastError();
   if(err!=cudaSuccess)printf("error: %d %s",err,cudaGetErrorString(err));  
    
    // Copy array back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
 
    // Sum up vector c and print result divided by n, this should equal 1 within error
    int sum = 0;
    for(i=0; i<10; i++)
        //sum += h_c[i];
    printf("final result: %d h_a=%d h_b=%d h_c=%d\n", i,h_a[i],h_b[i], h_c[i]);
 
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}
