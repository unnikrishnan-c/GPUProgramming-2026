
#include<cuda.h>
#include<stdio.h>
__device__ float sum;
__global__ void K1(){
	int var=0;//thread private  variable.
	if(threadIdx.x%4==0) {
		printf("thread id==%d \n", threadIdx.x);
		var=100;
	}
	if(var&&threadIdx.x >=16)var=200;
	__syncthreads();
	int bits=__ballot_sync(0xFFFFFFFF,var==100);//predicate true for threads 0,4,8,12,16,20,24,28
	if(threadIdx.x==0)printf("count=%d \n", __popc(bits));
	int x= __shfl_sync(0xF0,var,16,8);//return value UNDEFINED for nonparticipating threads, as per the CUDA manual. 
       if(threadIdx.x <=31 && x==200)printf("tid=%d var= %d x=%d \n",threadIdx.x,var,x);	
       if(threadIdx.x <=31 && x==100)printf("tid=%d var= %d x=%d \n",threadIdx.x,var,x);	
}
main(){
	K1<<<1,32>>>();
	cudaDeviceSynchronize();

}
