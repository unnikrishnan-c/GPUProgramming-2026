#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>
#include <iostream>
#include <iterator> //for std::ostream_iterator

int main(int argc, char *argv[]) {
    //Allocate vector on host
       thrust::host_vector<int> h_vec(10);//vector of 10 elements on host/CPU
       int n=atoi(argv[1]);// random number upper limit
       //initialize the vector
       thrust::generate(h_vec.begin(), h_vec.end(), [n]() {
        return (rand() % n) + 1;
    });
       //print the vector
    std::cout << "Initial vector: ";
    thrust::copy(h_vec.begin(), h_vec.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout <<"\n";
    // copy vector from host/CPU to device/GPU 
    thrust::device_vector<int> d_vec = h_vec;

    // 3. Sort vector on device.
    thrust::sort(d_vec.begin(), d_vec.end());

    // 4. copy sorted vector from device to  host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

    // 5. Print all elements from the host vector
    std::cout << "Sorted vector: ";
    thrust::copy(h_vec.begin(), h_vec.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout <<"\n";

    return 0;
}

