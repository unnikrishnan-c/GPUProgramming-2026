#include <thrust/host_vector.h>
    #include <thrust/device_vector.h>
    #include <thrust/sort.h>
    #include <iostream>

    int main() {
        // Example using Thrust
        thrust::host_vector<int> h_vec(5);
        // ... populate h_vec
        thrust::device_vector<int> d_vec = h_vec;
        thrust::sort(d_vec.begin(), d_vec.end());
        // ... copy back to host or continue processing on device
        return 0;
    }
