#include "dmr_kernels.h"

void matrix_mult_dmr(float *A, float *B, int M, int N, int K, float *C, int layerIndex) {
    // printf("layer %d: [%dx%d] X [%dx%d] = [%dx%d]\n", layerIndex, M, K, K, N, M, N);
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    matrix_mult_dmr_kernel<1, 100><<<dimGrid,dimBlock>>>(A, B, M, N, K, C, layerIndex);
    // check_error(cudaError_t(cudaPeekAtLastError()));
}