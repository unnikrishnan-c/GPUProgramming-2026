#include<stdio.h>
#include<cuda.h>
__device__ int changed;
__global__ void K1(){
	__shared__ int s;
	if(threadIdx.x==0){
		s=0;
		printf("hello %d\n",blockIdx.x);
	}
	if(threadIdx.x==0 && blockIdx.x==1)s=s+6;//warp0
	if(threadIdx.x==48&&blockIdx.x==1)s=s+5;//warp1
	if(threadIdx.x==10 )s=s+4;
	if(threadIdx.x==0 &&blockIdx.x==1)changed=s;
	
}
int main() {
	K1<<<2,64>>>();
	cudaDeviceSynchronize();
	int temp;
	cudaMemcpyFromSymbol(&temp,changed,sizeof(int),0,cudaMemcpyDeviceToHost);
	printf("temp=%d\n",temp);
}
