#include <cuda.h>
#include <iostream>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__ )

void check(CUresult result, const char *func, const char *file, int line) {
    if (result != CUDA_SUCCESS) {
	    printf("CUDA Error %d\n",line); 
        exit(1);
    }
}

int main(int argc, char *argv[]) {
	      checkCudaErrors(cuInit(0));

    // Get the first CUDA device
    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));

    // Create a context
    CUcontext context;
    checkCudaErrors(cuCtxCreate(&context, 0, device));

    // Load the cubin file
    CUmodule module;
    checkCudaErrors(cuModuleLoad(&module, argv[1]));

    // Get the kernel function from cubin file.
    CUfunction kernel;
    //checkCudaErrors(cuModuleGetFunction(&kernel, module, "K1"));
    checkCudaErrors(cuModuleGetFunction(&kernel, module, argv[2]));

    void *kernelParams[] = { };

    checkCudaErrors(cuLaunchKernel(
        kernel,
        1, 1, 1,   // grid_size 
        32, 1, 1,   // block_size
        0, nullptr,  kernelParams, nullptr ));

    checkCudaErrors(cuCtxSynchronize());

    printf("Kernel executed successfully.");
   
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}

