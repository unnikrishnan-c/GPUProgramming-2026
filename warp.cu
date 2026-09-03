#include<cuda.h>
#include<stdio.h>
#include<unistd.h>
// Allocate an interger variable statically on device/GPU
__device__ int val = 32;
int *ptr;
// K1: Function executed on device/GPU, and called from host/CPU
__global__ void K1 (int *dptr, int N)
{   
	int tid=blockIdx.x *blockDim.x+threadIdx.x;

  if(blockIdx.x==0)atomicAdd(&val,1);
  if(tid %32==0 ){
	  __syncthreads();//first thread in each warp satisfy the condition.
  }
 // __syncthreads();//all threads
  if(blockIdx.x==0 && val < 1056)printf ("threadIdx.x =%d blockIdx.x=%d , val=%d\n",threadIdx.x,blockIdx.x, val);
//  dptr[threadIdx.x] = threadIdx.x;
}

int main ()
{//to find errors when kernel call finishes.
  cudaError_t err;
  int *dptr;
  //allocate memory on device.
  cudaMalloc ((void **) &dptr, sizeof (int) * 10);
  err = cudaGetLastError ();
  K1 <<<1 , 1024 >>> (dptr,30);
  cudaDeviceSynchronize ();
  err = cudaGetLastError ();
  if (err != cudaSuccess)
    printf ("error in kernel call");
/*  K1 <<< 1, 1 >>> (dptr);
  cudaDeviceSynchronize ();
  int temp;
  int *hptr;
  hptr = (int *) malloc (sizeof (int) * 10);
  if(cudaMemcpyFromSymbol(&temp, val, sizeof (int), 0,cudaMemcpyDeviceToHost) != cudaSuccess) printf ("memcpy error\n");
  if (cudaMemcpy (hptr, dptr, sizeof (int) * 10, cudaMemcpyDeviceToHost) !=cudaSuccess)printf ("memcpy error\n");
  printf ("host: %d\n", temp);
  for (int i = 0; i < 10; i++)
    printf (" %d \n", hptr[i]);
  return 0;*/
}
