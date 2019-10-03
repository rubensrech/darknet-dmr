#ifndef GEMM_KERNELS_H_
#define GEMM_KERNELS_H_

// #ifdef __cplusplus
// extern "C" {
// #endif

void matrix_mult_dmr(float *A, float *B, int M, int N, int K, float *C, int layerIndex);

// #ifdef __cplusplus
// }
// #endif

#endif /* GEMM_KERNELS_H_ */