#include<stdio.h>
#include<stdlib.h>
//Number of SMs=68, Number of SPs in an SM=64
__device__ int counter;//initialized to zero
__global__ void barrier (int x) {

        //x-> number of blocks --> maximum value for x is number of SMs
        int val=x;//number of blocks

        /*Computation before barrier for kernel/grid*/
        if(threadIdx.x ==0 )  atomicAdd(&counter,1);//thread zero in each block increment counter by one
      if(threadIdx.x==0) printf("before barrier block id=%d  counter=%d\n", blockIdx.x, counter);
           __syncthreads();//barrier for one block,1024 threads in a block
                while(atomicCAS(&counter,val,val)!=val);//loop until counter has become number of blocks.
        __syncthreads();//barrier for one block.
        /*Computation after barrier for grid/kernel*/
      if(threadIdx.x==0) printf("block id=%d  counter=%d\n", blockIdx.x, counter);

}


int main(int argc, char *argv[]){
        int X=atoi(argv[1]);
        printf("begin");
        barrier <<< X, 32 >>> (X);
        cudaDeviceSynchronize();
        printf("end");
        return 0;
}

