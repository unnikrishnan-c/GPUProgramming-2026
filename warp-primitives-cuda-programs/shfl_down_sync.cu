#include<stdio.h>
__global__ void K1(){
	int tid=threadIdx.x;
	int val=10;//thread private variable.
         int offset=1;
	for (int offset = 16; offset > 0; offset /= 2){//16,8,4,2,1
		//warp coopeative communication
    val += __shfl_down_sync(0x0, val, offset/*delta*/);//fetch value from a thread having a higher thread id than caller.
	if(threadIdx.x==0)printf("tid=%d val=%d \n",tid,val);
	if(threadIdx.x==2)printf("tid=%d val=%d \n",tid,val);
	if(threadIdx.x==4)printf("tid=%d val=%d \n",tid,val);
	if(threadIdx.x==8)printf("tid=%d val=%d \n",tid,val);
	}
//	printf(" threadid=%d val=%d\n",threadIdx.x, val);

	val=1;

//	for (int offset = 1; offset < 32; offset *= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	//printf(" threadid=%d val=%d\n",threadIdx.x, val);
}
main(){
	cudaError_t err=cudaGetLastError();
	K1<<<1,32>>>();
	cudaDeviceSynchronize();
	err=cudaGetLastError();
	if(err!=cudaSuccess)printf("%s",cudaGetErrorString(err));
}
