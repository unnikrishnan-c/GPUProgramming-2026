#include<stdio.h>
#include<stdlib.h>
//Number of SMs=68, Number of SPs in an SM=64
__device__ int counter;//initialized to zero
__global__ void barrier (int x , int *addr,int stream_id) {//x-> number of blocks --> maximum value for x is number of SMs
	int val=x;//number of blocks
   
	/*Computation before barrier for kernel/grid*/
	if(threadIdx.x ==0 )  atomicAdd(addr,1);//thread zero in each block increment counter by one
	   __syncthreads();//barrier for one block,1024 threads in a block
		while(atomicCAS(addr,val,val)!=val);//loop until counter has become number of blocks.
        __syncthreads();//barrier for one block.
	/*Computation after barrier for grid/kernel*/
      if(threadIdx.x==0 &&blockIdx.x==67) printf("block id=%d  counter=%d stream id=%d\n", blockIdx.x, *addr,stream_id);

}


int main(){
        int X=68;
	printf("begin");
	int *x1,*x2,*x3;
	cudaMalloc(&x1,sizeof(int));
	cudaMalloc(&x2,sizeof(int));
	cudaMalloc(&x3,sizeof(int));
	int temp=0;
	cudaMemcpy(x1,&temp,sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(x2,&temp,sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(x3,&temp,sizeof(int),cudaMemcpyHostToDevice);
cudaStream_t s1, s2, s3;
int s1id=0,s2id=1,s3id=2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);
cudaStreamCreate(&s3);
	barrier <<< X, 1024,0,s1  >>> (X,x1, s1id);
	printf("done stream one\n");
	barrier <<< X, 1024,0,s2  >>> (X,x2,s2id);
	printf("done stream two\n");
	barrier <<< X, 1024,0,s3  >>> (X,x3,s3id);
	printf("done stream three\n");
	cudaDeviceSynchronize();
	printf("end");
	return 0;
}
