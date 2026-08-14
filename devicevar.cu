#include<cuda.h>
#include<stdio.h>
#include<unistd.h>
// Allocate an interger variable statically on device/GPU
__device__ int val=32;
int *ptr;
__device__ int DA[20][10];

__global__ void init (){
	int *ptr=&DA[0][0];//referece to  first element in the matrix DA.
		for(int i=0;i<20;i++)
		for (int j=0;j<10;j++) ptr[i*10+j]=i*10+j;//DA[i][j]=i*10+j
		}
// K1: Function executed on device/GPU, and called from host/CPU
__global__ void K1(int *dptr ){
       
	printf("x=%d y=%d val=%d\n", threadIdx.x, threadIdx.y, DA[threadIdx.x][threadIdx.y]);
}

int main(){
	//cudaError_t datatype: To catch errors in  CUDA kernel calls.
	cudaError_t err;
	//pointer variable can be used to allocate memory on host using malloc() and on device using cudaMalloc.
	int *dptr;
	//allocate memory on device/GPU using cudaMalloc: array of 10 integers
	cudaMalloc((void **)&dptr,sizeof(int)*100);
      init<<<1,1>>>();
      cudaDeviceSynchronize();
      int *hptr;
      hptr=(int *)malloc(sizeof(int)*100);
      for(int i=0;i<100;i++)hptr[i]=i;
      if(cudaMemcpy(dptr, hptr,sizeof(int)*100,cudaMemcpyHostToDevice)!=cudaSuccess)printf("memcpy error\n");
	//before the kernel call, flush previous error by calling cudaGetLastError().
	err=cudaGetLastError();
	dim3 tpb(10,20);
      K1<<<1,tpb>>>(dptr);
      cudaDeviceSynchronize();
      //now check last error,if there it will be from call to K1.
	err=cudaGetLastError();
	
	//if err is not cudaSuccess, kernel call had an issue.
	if(err!=cudaSuccess)printf("error in kernel call");
      K1<<<1,1>>>(dptr);
      cudaDeviceSynchronize();
      int temp;

      //allocate array of 10 integers on host/CPU using malloc.
      //copy statically allocate device variable from device/GPU to host/CPU using cudaMemcpyFromSymbol() function.
      if(cudaMemcpyFromSymbol(&temp, val,sizeof(int),0,cudaMemcpyDeviceToHost)!=cudaSuccess)printf("memcpy error\n");
      //copy result from device/GPU to host/CPU using cudaMemcpy() function :   hptr[0-9]=dptr[0-9]
      if(cudaMemcpy(hptr, dptr,sizeof(int)*10,cudaMemcpyDeviceToHost)!=cudaSuccess)printf("memcpy error\n");
	printf("host: %d\n", temp);
	for(int i=0;i<10;i++)printf(" %d \n", hptr[i]);
	return 0;
}

