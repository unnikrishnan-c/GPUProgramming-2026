#include <cstdio>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace wmma;
// size of the tile (for simplicity use 16×16×16)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void init(half *A,half *B){
 int tid=blockIdx.x * blockDim.x + threadIdx.x;
 A[tid]=tid;
 B[tid]=tid;
}

// A: (M×K) ; B: (K×N) ; C: (M×N)
__global__ void tensorCoreGemmKernel(half *A, half *B, float *C, int M, int N, int K) {
   // each warp computes one tile of output C
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;// warps per block is one, 
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);//zero for this example for 16x16 input matrix, called kernel with y dimension being one.
    int mTile = warpM * WMMA_M;
    int nTile = warpN * WMMA_N;//value is zero.
   if(warpN!=0)printf(" %d \n", warpN);
   if( threadIdx.x==0)printf(" %d \n", warpM);
    if (mTile >= M || nTile >= N) return;

    // Declare the fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> aFrag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> bFrag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    // Initialize the output to zero
    fill_fragment(cFrag, 0.0f);
    //Compute C(1,1) of the 32x32 matrix.
        for (int i=0;i<2;i++){
               	int bind=16+16*K*i;//i=0-> 16, i=1 -> 16x(1+32) (0,1) ,  (1,1)
                int aind=i*16+M*16;//i=0->32*16, i=1 --> 16+32*16  (1,0) , (1,1)
//if(threadIdx.x==0)		printf("i=%d bind=%d aind=%d \n", i, bind,aind);
           load_matrix_sync(aFrag, A +aind,K);//load submatrix A  into registers
            load_matrix_sync(bFrag, B + bind,N);//load submatrix B into registers
            mma_sync(cFrag, aFrag, bFrag, cFrag);//do the multiplication.
         }
// writting result in to first element of tile C(0,0)//incorrect
   store_matrix_sync(C, cFrag, N, mem_row_major);
 //index in to tile C(1,1)
   float *loc= C+32*16+16;//starting address of the tile C(1,1), element at 32nd row, 16th column
// writting result in to first element of tile C(1,1).correct
   store_matrix_sync(loc, cFrag, N, mem_row_major);
}

int main() {
	//half : one bit for sign, 5 bit for exponent, Fraction/Mantisa: 10 bits
	//single: one bit for sign, 8 bits for exponent, 23 bits for fraction.
    int M = 32, N = 32, K = 32;  // multiple of 16 expected by GPU tensor core.
    half *devA;
    half *devB;
    float *devC,*hostC;
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&devA, M * K * sizeof(half));
    cudaMalloc(&devB,K * N * sizeof(half));
    cudaMalloc(&devC, M * N * sizeof(float));
    hostC=(float *)malloc(M*N*sizeof(float));

    cudaError_t    err=cudaGetLastError();
    init<<<1,1024>>>(devA, devB);
    cudaDeviceSynchronize();
    err=cudaGetLastError();
     if(err!=cudaSuccess) printf("%s\n", cudaGetErrorString(err));
    err=cudaGetLastError();
    //code computes C(1,1) and stores it in C(0,0) and C(1,1), see two calls to store_matrix_sync inside the CUDA kernel
    cudaEventRecord(start, 0);//starttime recorded.
    tensorCoreGemmKernel<<<1, 32>>>(devA, devB, devC, M, N, K);//one warp, computes c(1,1)
    cudaEventRecord(stop, 0);//endtime recoorded
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %f milli seconds\n", elapsedTime);
    err=cudaGetLastError();
    if(err!=cudaSuccess) printf("%s\n", cudaGetErrorString(err));
    cudaMemcpy(hostC,devC,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    fprintf(stderr, "last two elemeents in C(0,0) %f %f\n", hostC[15*32+14],hostC[15*32+15]);
    fprintf(stderr, "last two elemeents in C(1,1) %f %f\n", hostC[1022],hostC[1023]);
    int res=0;
    int  startrow=32*31;
    int startcol= 30;
    for(int i=0;i<M;i++)res+=(startrow+i)*(startcol+i*32);
    printf("expected result %d\n", res);
    res=0;
      startrow=32*31;
     startcol= 31;
    for(int i=0;i<M;i++)res+=(startrow+i)*(startcol+i*32);
    printf("expected result %d\n", res);
    //check C(0,0) and C(1,1) is the same. Look at two writes to tiles C(0,0) and C(1,1) using store_matrix_sync() in the kernel tensorCoreGemmKernel.
    float *start_tile_zero=hostC;
    float *start_tile_four=&hostC[32*16+16];
    int count=0;
    //C(0,0) and C(1,1) should be same. Checking the same below. 
    //total 512 elements should be nonzero. value of variable count will be only 256.
    for(int i=0;i<16;i++){
        for(int j=0;j<16;j++){
	    if(start_tile_zero[i*32+j]!=start_tile_four[i*32+j])printf("Error: i=%d", i);
	    if(start_tile_zero[i]==start_tile_four[i])count++;
	    }
    }
printf("Count=%d \n",count);
    return 0;

}


//for a matrix of size 64x64. We have 16 tiles of size 16x16.
//(0,0) (0,1) (0,2) (0,3)
//(1,0) (1,1) (1,2) (1,3)
//(2,0) (2,1) (2,2) (2,3)
//(3,0) (3,1) (3,2) (3,3)
//starting address of the tiles
//(0,0) - 0 , (0,1) - 16, (0,2)-32, (0,3)-48
//(1,0) - X , (1,1) - X+16 , (1,2) - X+32, (1,3)-X+48
//(2,0) - Y , (2,1) - Y+16 , (2,2) - Y+32, (2,3)-Y+48
//(3,0) - Z , (3,1) - Z+16 , (3,2) - Z+32, (3,3)-Z+48
//X=64x16, Y=64x32, Z= 64x48
//warp-id/4+warp-id*16

// (0,0) (0,1)       (0,0) (0,1)
// (1,0) (1,1)       (1,0) (1,)
