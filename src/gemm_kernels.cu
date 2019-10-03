#include "stdint.h"

// extern "C" {
    #include "cuda.h"
// }

#define ZERO_FLOAT 2.2e-20
#define ZERO_DOUBLE 1.4e-40
#define ZERO_HALF 4.166e-05

#define BLOCK_SIZE 16

#define THRESHOLD 1
#define CHECK_BLOCK 100

#ifdef GPU

#define LAYERS 107

typedef struct {
    unsigned long long errors;
} LayerErrors;

__device__ LayerErrors errorsPerLayer[LAYERS] = {0};

__device__ __forceinline__ void axpy__(const double a, const double b, double &c) {
    c = __fma_rn(a, b, c);
}
__device__ __forceinline__ void axpy__(const float a, const float b, float &c) {
    c = __fmaf_rn(a, b, c);
}
__device__ __forceinline__ void axpy__(const double a, const double b, float &c) {
    c = __fmaf_rn(__double2float_rn(a), __double2float_rn(b), c);
}
__device__ __forceinline__ void axpy__(const float a, const float b, __half &c) {
    c = __hfma(__float2half(a), __float2half(b), c);
}

__device__ int check_bit_error(const __half &lhs, const float &rhs) {
	const uint32_t lhs_data = __float_as_uint(__half2float(lhs));
	const uint32_t rhs_data = __float_as_uint(rhs);
	uint32_t sub_res;
	if (lhs_data > rhs_data) {
		sub_res = lhs_data - rhs_data;
	} else {
		sub_res = rhs_data - lhs_data;
	}

	if (sub_res > THRESHOLD) {
        return 1;
	} else {
        return 0;
    }
}

__device__ int check_bit_error(const float &lhs, const float &rhs) {
	float diff = fabs(lhs - rhs);
	if (diff > ZERO_FLOAT) {
        return 1;
	} else {
        return 0;
    }
}


__global__ void matrix_mult_dmr_kernel(float *A, float *B, int M, int N, int K, float *C, int layerIndex) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        register float acc_real_t = 0.0;
	    register __half acc_half_t = 0.0;

        // #pragma unroll 10
        for (int i = 0; i < K; i++) {
            axpy__(A[row * K + i], B[i * N + col], acc_real_t);
            // axpy__(A[row * M + i], B[col * N + i], acc_half_t);

            // if ((i % CHECK_BLOCK) == 0) {
            //     if (check_bit_error(acc_half_t, acc_real_t)) {
            //         atomicAdd(&errorsPerLayer[layerIndex].errors, 1);
            //     }
            //     acc_half_t = __half(acc_real_t);
            // }
        }

        C[row * N + col] = acc_real_t;
    }

}

void matrix_mult_dmr(float *A, float *B, int M, int N, int K, float *C, int layerIndex) {
    printf("layer %d: [%dx%d] X [%dx%d] = [%dx%d]\n", layerIndex, M, K, K, N, M, N);
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    matrix_mult_dmr_kernel<<<dimGrid,dimBlock>>>(A, B, M, N, K, C, layerIndex);
	// check_error(cudaError_t(cudaPeekAtLastError()));
}

#endif