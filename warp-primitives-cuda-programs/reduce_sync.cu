#include<stdio.h>
__global__ void K1(){
	int tid=threadIdx.x;
	int val=10;
	//causing compile time error. Why?
	int addval=__reduce_add_sync(0xFFFFFFFF,val);
	if(threadIdx.x==0)printf("Val=%d addval=%d\n",val,addval);
}
int main(){
	K1<<<1,32>>>();
	cudaDeviceSynchronize();
       return 0;
}
