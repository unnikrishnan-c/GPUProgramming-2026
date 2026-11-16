#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <thrust/copy.h>
#include <vector>
#include <iostream>
#include <iterator>

// Functor 
struct MatMulFunctor {
    const int* A;
    const int* B;
    int K;
    int N;
    MatMulFunctor(const int* _A, const int* _B, int _K, int _N)
        : A(_A), B(_B), K(_K), N(_N) {}

    __host__ __device__
    int operator()(int idx) const {
        int row = idx / N;
        int col = idx % N;
        int sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        return sum;
    }
};

int main() {
    int M = 2; // Rows in A and C
    int K = 3; // Columns in A, Rows in B
    int N = 2; // Columns in B and C

    std::vector<int> h_A = {1, 2, 3, 4, 5, 6};//2x3 matrix
    std::vector<int> h_B = {7, 8, 9, 10, 11, 12};//3x2 matrix

    thrust::device_vector<int> d_A = h_A;//copy A from host to device
    thrust::device_vector<int> d_B = h_B;//copy B from host to device
    thrust::device_vector<int> d_C(M * N);//vector of size 2*2=4

    const int* A_ptr = thrust::raw_pointer_cast(d_A.data());//get pointer to device vector A
    const int* B_ptr = thrust::raw_pointer_cast(d_B.data());//get pointer to device vector B

    thrust::tabulate(d_C.begin(), d_C.end(), MatMulFunctor(A_ptr, B_ptr, K, N));//3,2

    std::cout << "Result Matrix C:\n";
    thrust::copy(d_C.begin(), d_C.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "\n";

    return 0;
}

