
#include<cuda.h>
#include<stdio.h>
__global__ void K1(){
	int var=0;//thread private  variable.
	if(threadIdx.x%4==0) {
		var=100;
	unsigned int active=__activemask();
	if(threadIdx.x==0){
	printf("inside if active=%u \n", active);
		printf("active threads: ");
	for(int i=0;i<32;i++)if(active & (0x1<<i))printf("%d ",i);
	printf("\n");
	}
	}//0,4,8,12,16,20,24,28
	unsigned int active=__activemask();
	if(threadIdx.x==0)printf("ouside if active=%u\n",active); 
	if(threadIdx.x==0){
		printf("active threads: ");
	for(int i=0;i<32;i++)if(active & (0x1<<i))printf(" %d ",i);
	printf("\n");
	}
}

main(){
	K1<<<1,32>>>();
	cudaDeviceSynchronize();
}
