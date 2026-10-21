#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#define N 1024
int *hArr;
//initializes an array to zero
__global__ void K1(int *dArr){
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
         dArr[tid]=0;


}
        //initialize array to threadid
__global__ void K2(int *dArr){
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
         dArr[tid]=tid;



}


int main(){
        int *dArr,*dArr1;
        /*trying to enable peer access, if hardware supports it, it will work*/
   cudaSetDevice(0);
  cudaDeviceEnablePeerAccess(1, 0);
  cudaSetDevice(1);
  if(cudaDeviceEnablePeerAccess(0, 0)!=cudaSuccess)printf("Peer Access Not Possible\n");
  //current Device is GPU1
        cudaSetDevice(0);
        //current device is GPU0
        cudaMalloc((void **)&dArr,sizeof(int)*1024);//allocate 1024 elements in GPU0
 cudaSetDevice(1);
        cudaMalloc((void **)&dArr1,sizeof(int)*1024);//allocate 1024 elements in GPU0
        hArr=(int *)malloc(sizeof(int)*1024);
//      dArr[10]=0;

        cudaSetDevice(1);
        cudaError_t err=cudaGetLastError();
cudaError_t err=cudaGetLastError();
// cudaSetDevice(1);
//K1 executes on GPU1
        //initialize array to zero
        K1<<<32, 32>>>(dArr);//kernel call put in stream zero of device
        cudaDeviceSynchronize();
         err=cudaGetLastError();
         if(err!=cudaSuccess){
                 printf("error K1: %s \n",cudaGetErrorString(err));
        }
        //Transfer from Device 1 to Device 0, current device is 0.
        if(cudaMemcpy(dArr,dArr1,sizeof(int)*1024,cudaMemcpyDeviceToDevice)!=cudaSuccess)printf("Memcpy Error: 1 to 0\n");
        //Transfer from Device 0 to Device 1, current device is 0.
        if(cudaMemcpy(dArr1,dArr,sizeof(int)*1024,cudaMemcpyDeviceToDevice)!=cudaSuccess)printf("Memcpy Error: 0 to 1\n");
        //initialize array to threadid
        //K2<<<32,32>>>(dArr);//kernel call put in the stream zero of device
        //else{                  
        cudaMemcpy(hArr,dArr,sizeof(int)*1024,cudaMemcpyDeviceToHost);//cudamemcpy put in stream zero if device
//      for(int i=0;i<1024;i++)printf("%d ",hArr[i]); 
        //for(int i=0;i<1024;i++)printf("%d ",hArr[i]); 
        //copy device array to host and print the array in host
        //}
        return 0;
}
                     
