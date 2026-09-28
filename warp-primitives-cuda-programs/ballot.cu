#include<cuda.h>
#include<stdio.h>
__device__ float sum;
__global__ void K1(){
	int var=0;//thread private  variable.
	if(threadIdx.x%4==0) {
		printf("thread id==%d \n", threadIdx.x);
		var=1;
	}
	//0,4,8,12,16,20,24,28-8
	__syncthreads();
	int bits=__ballot_sync(0xFFFFFFFF,var==1);//predicate true for threads 0,4,8,12,16,20,24,28
	printf("count=%d\n", __popc(bits));
	__syncthreads();
	
	int x= __shfl_sync(0xFFFFFFFF,var,4); 
	if(x!=1)printf("error\n");
	int y= __shfl_sync(0xFFFFFFFF,var,3); 
	if(y==1)printf("error\n");
        int z=__ffs(bits);
	if(threadIdx.x==z){
		printf("set=%d  %d %d \n",threadIdx.x,__ffs(0),__ffs(1));
	}

	/*bits=bits^z;
         z=__ffs(bits);
	if(threadIdx.x==z)printf("set=%d \n",threadIdx.x);
	bits=bits^(1<< (z-1));
         z=__ffs(bits);
	if(threadIdx.x==z)printf("set=%d \n",threadIdx.x);
	bits=bits^(1<< (z-1));
         z=__ffs(bits);
	if(threadIdx.x==z)printf("set=%d \n",threadIdx.x);

	}*/
	}
main(){
K1<<<1,32>>>();
cudaDeviceSynchronize();
}




