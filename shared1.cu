#include<cuda.h>
#include<stdio.h>

//#define SIZE 65536
//#define SIZE 32768
#define SIZE 49152
#define BLOCKSIZE 32
__global__ void dkernel() {
__shared__ unsigned char data[SIZE];
data[threadIdx.x] = threadIdx.x;
printf("%d\n", data[threadIdx.x]);
}
int main() {
cudaFuncSetCacheConfig(dkernel, cudaFuncCachePreferL1);
//cudaFuncSetCacheConfig(dkernel, cudaFuncCachePreferShared);
//int	carveout=50;
//cudaFuncSetAttribute(dkernel, cudaFuncAttributePreferredSharedMemoryCarveout,carveout);
cudaError_t err;
err=cudaGetLastError();
dkernel<<<1, BLOCKSIZE>>>();
cudaDeviceSynchronize();
err=cudaGetLastError();
if(err!=cudaSuccess)fprintf(stderr, "Error : %s", cudaGetErrorString(err));
}
