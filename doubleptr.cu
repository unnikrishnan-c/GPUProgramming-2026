#include<cuda.h>
#include<stdio.h>
struct node {
        int *dist;//allocate on devie
        float *xx;
}*head;//allocate on device
__global__ void check(int *dist, float *xx){
        printf(" check: %d %f\n",dist[0],xx[0]);
}
__global__ void check1( struct node *temp){
        int *ptr1=temp->dist;
        float *ptr2=temp->xx;
if(ptr1!=NULL &&ptr2!=NULL)     printf(" check: %d %f\n",ptr1[0],ptr2[0]);
}
int main(){
        cudaError_t err;
//temp to hold device address.
        struct node temp;
        //head has address in device/GPU
        if(cudaMalloc(&head,sizeof(struct node))!=cudaSuccess)printf("error allocating head");
        //copy device address value to host/CPU
        cudaMemcpy(&temp,head,sizeof(struct node),cudaMemcpyDeviceToHost);
        //allocte dist and xx field of head on devide/GPU. 
if(cudaMalloc(&(temp.dist),sizeof(int)*1024)!=cudaSuccess)printf("error allocating dist");
if(cudaMalloc(&(temp.xx),sizeof(float)*1024)!=cudaSuccess)printf("error allocating dist");

        //below statement will not work: not possible to dereference device pointer from host like head->dist
        //if(cudaMalloc(&(head->dist),sizeof(int)*1024)!=cudaSuccess)printf("error allocating dist");

//copy the temp with dist and xx fields having device address to head
        cudaMemcpy(head,&temp, sizeof(struct node),cudaMemcpyHostToDevice);
        int var=1025;
        float tt=2.5;
        
//copy one element to the zeroth position  of head->dist and head->xx on device/GPU
        cudaMemcpy(temp.dist, &var,sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(temp.xx, &tt,sizeof(float), cudaMemcpyHostToDevice);
        err=cudaGetLastError();
        check1<<<1,1>>>(head);
        cudaDeviceSynchronize();
        err=cudaGetLastError();
        if(err!=cudaSuccess)printf("%s", cudaGetErrorString(err));
        return 0;
}

