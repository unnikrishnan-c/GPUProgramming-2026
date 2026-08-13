#include <iostream>
#include <cuda_runtime.h>
int getCoresPerSM(int major, int minor) {
    // Defines cores per SM based on architecture generation
    switch (major) {
 case 2: // Fermi
            return (minor == 1) ? 48 : 32;
        case 3: // Kepler
            return 192;
        case 5: // Maxwell
            return 128;
        case 6: // Pascal
            if (minor == 1 || minor == 2) return 128;
            if (minor == 0) return 64;
            return 128; // Default fallback for Pascal
        case 7: // Volta (7.0), Turing (7.5)
            return 64;
        case 8: // Ampere (8.0, 8.6, 8.7), Ada Lovelace (8.9)
            if (minor == 0) return 64;
            if (minor == 6 || minor == 9) return 128;
            return 64; // Default fallback for Ampere variants
        case 9: // Hopper (9.0), Blackwell (9.5)
            return 128;
        default:
            return 128; // Standard fallback for future architectures
    }
}
int main() {
    int deviceCount = 0;
    
    // Get the total number of CUDA-enabled devices
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s).\n" << std::endl;

    // Loop through each available device
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        
        // Populate the property structure for the current device index
        cudaGetDeviceProperties(&prop, i);

        std::cout << "--- Device " << i << ": " << prop.name << " ---" << std::endl;
        std::cout << "  Compute Capability:          " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory:         " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Streaming Multiprocessors:   " << prop.multiProcessorCount << std::endl;
        std::cout << "  Cores Per SM:                " << getCoresPerSM(prop.major, prop.minor) << std::endl;
        std::cout << "  Total Cores:                 " << prop.multiProcessorCount*getCoresPerSM(prop.major, prop.minor) << std::endl;
        std::cout << "  Max Threads Per Block:       " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Shared Memory Per Block :     " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Warp Size:                   " << prop.warpSize << std::endl;
        std::cout << std::endl;
    }

    return 0;
}

