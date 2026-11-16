#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error at line " << __LINE__ << ": " \
                  << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSPARSE(func) { \
    cusparseStatus_t status = (func); \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "cuSPARSE Error at line " << __LINE__ << ": " \
                  << cusparseGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    const int64_t rows = 4;
    const int64_t cols = 4;
    const int64_t nnz  = 8; // Total number of non-zero elements

    // CSR representation of matrix A
    std::vector<int>   h_csr_offsets = { 0, 2, 3, 6, 8 };
    std::vector<int>   h_csr_columns = { 0, 2, 1, 0, 2, 3, 1, 3 };
    std::vector<float> h_csr_values  = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };

    //Input vector x and output vector y
    std::vector<float> h_x = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<float> h_y(rows, 0.0f);

    //Scalars for y = alpha * A * x + beta * y
    float alpha = 1.0f;
    float beta  = 0.0f;

    //Allocate device memory
    int   *d_csr_offsets = nullptr;
    int   *d_csr_columns = nullptr;
    float *d_csr_values  = nullptr;
    float *d_x           = nullptr;
    float *d_y           = nullptr;

    CHECK_CUDA(cudaMalloc(&d_csr_offsets, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_columns, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_values,  nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x,           cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y,           rows * sizeof(float)));

    // 2. Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_csr_offsets, h_csr_offsets.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_columns, h_csr_columns.data(), nnz * sizeof(int),        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_values,  h_csr_values.data(),  nnz * sizeof(float),      cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x,           h_x.data(),           cols * sizeof(float),     cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y,           h_y.data(),           rows * sizeof(float),     cudaMemcpyHostToDevice));

    // 3. Initialize cuSPARSE and create descriptors
    cusparseHandle_t     handle = nullptr;
    cusparseSpMatDescr_t matA   = nullptr;
    cusparseDnVecDescr_t vecX   = nullptr;
    cusparseDnVecDescr_t vecY   = nullptr;

    CHECK_CUSPARSE(cusparseCreate(&handle));

    //Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, rows, cols, nnz,
        d_csr_offsets, d_csr_columns, d_csr_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Create dense vectors X and Y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, rows, d_y, CUDA_R_32F));

    //Query workspace size and allocate buffer
    size_t bufferSize = 0;
    void*  dBuffer    = nullptr;

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    //Execute SpMV
    CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    //Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Result vector y: ";
    for (float val : h_y) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(d_csr_offsets));
    CHECK_CUDA(cudaFree(d_csr_columns));
    CHECK_CUDA(cudaFree(d_csr_values));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}

