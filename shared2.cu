#include <stdio.h>
#include <cuda.h>
__global__ void dynshared() {
extern __shared__ int s[];//__shared__ int s[n*sizeof(int)]
s[threadIdx.x] = threadIdx.x;
__syncthreads();
if (threadIdx.x % 2) printf("%d\n", s[threadIdx.x]);
}
int main() {
int n;
scanf("%d", &n);
int carveout=50;

cudaError_t err;
err=cudaGetLastError();
cudaFuncSetAttribute(dynshared, cudaFuncAttributePreferredSharedMemoryCarveout,carveout);
dynshared<<<1, 1024, n * sizeof(int)>>>();
cudaDeviceSynchronize();
err=cudaGetLastError();
if(err!=cudaSuccess)fprintf(stderr, "Error : %s", cudaGetErrorString(err));
return 0;
}
