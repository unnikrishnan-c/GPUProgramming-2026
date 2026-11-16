#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <thrust/copy.h>
#include <vector>
#include <iostream>
#include <iterator>

// Functor 
struct MatMulFunctor {
    const float* A;
    const float* B;
    int K;
    int N;
    MatMulFunctor(const float* _A, const float* _B, int _K, int _N)
        : A(_A), B(_B), K(_K), N(_N) {}

    __host__ __device__
    float operator()(int idx) const {
        int row = idx / N;
        int col = idx % N;
        float sum = 0.0f;
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

    std::vector<float> h_A = {1.1, 2.2, 3.3, 4.4, 5.5, 6.5};
    std::vector<float> h_B = {7.7, 8.8, 9.9, 10.0, 11.11, 12.12};

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(M * N);

    const float* A_ptr = thrust::raw_pointer_cast(d_A.data());
    const float* B_ptr = thrust::raw_pointer_cast(d_B.data());

    thrust::tabulate(d_C.begin(), d_C.end(), MatMulFunctor(A_ptr, B_ptr, K, N));

    std::cout << "Result Matrix C:\n";
    thrust::copy(d_C.begin(), d_C.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\n";

    return 0;
}

