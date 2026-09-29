#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
__device__ int counter;
__global__ void K1( ){
  atomicAdd(&counter,1);
  if(threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0){
	  printf("GridDim x=%d  y=%d z=%d\n", gridDim.x, gridDim.y, gridDim.z);
  printf("blockDim x=%d  y=%d z=%d\n", blockDim.x, blockDim.y, blockDim.z);
}

}

int main(){
int temp;
dim3 BPG(2,3,4);
//	dim3 BPG(2,3);
	dim3 TPB(5,6,30);
	cudaError_t err;
	err=cudaGetLastError();
	K1<<<BPG, TPB>>> ();
	cudaDeviceSynchronize();
	err=cudaGetLastError();
	if(cudaMemcpyFromSymbol(&temp,counter,sizeof(int))!=cudaSuccess)printf("error");
	printf("\ncounter=%d\n",temp);
	return 0;

}
