#ifndef DRM_KERNELS_H
#define DRM_KERNELS_H

#include "stdint.h"
#include "dmr_types.h"
#include "assert.h"

extern "C" { 
#include "cuda.h"
}

#define ZERO_FLOAT 2.2e-20
#define ZERO_DOUBLE 1.4e-40
#define ZERO_HALF 4.166e-05

#define BLOCK_SIZE 16

template<const uint32_t THRESHOLD_uint32_t>
__device__ int check_bit_error(const __half &lhs, const float &rhs) {
	const uint32_t lhs_data = __float_as_uint(__half2float(lhs));
	const uint32_t rhs_data = __float_as_uint(rhs);
	uint32_t sub_res;
	if (lhs_data > rhs_data) {
		sub_res = lhs_data - rhs_data;
	} else {
		sub_res = rhs_data - lhs_data;
	}

	if (sub_res > THRESHOLD_uint32_t) {
        return 1;
	} else {
        return 0;
    }
}

template<const uint32_t THRESHOLD_uint32_t>
__device__ int check_bit_error(const float &lhs, const float &rhs) {
	float diff = fabs(lhs - rhs);
	if (diff > ZERO_FLOAT) {
        return 1;
	} else {
        return 0;
    }
}

// > MxM

__device__ __forceinline__ void axpy__(const float a, const float b, float &c) {
    c = __fmaf_rn(a, b, c);
}
__device__ __forceinline__ void axpy__(const float a, const float b, __half &c) {
    c = __hfma(__float2half(a), __float2half(b), c);
}

template<const uint32_t THRESH, const uint32_t COUNT>
__global__ void matrix_mult_dmr_kernel(float *A, float *B, int M, int N, int K, float *C, unsigned long long *errorsCount) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        register float acc_real_t = 0.0;
	    register __half acc_half_t = 0.0;

        #pragma unroll COUNT
        for (int i = 0; i < K; i++) {
            axpy__(A[row * K + i], B[i * N + col], acc_real_t);
            axpy__(A[row * K + i], B[i * N + col], acc_half_t);

            if ((i % COUNT) == 0) {
                if (check_bit_error<THRESH>(acc_half_t, acc_real_t)) {
                    atomicAdd(errorsCount, 1);
                }
                acc_half_t = __half(acc_real_t);
            }
        }

        C[row * N + col] = acc_real_t;
    }

}

extern "C" void matrix_mult_dmr(float *A, float *B, int M, int N, int K, float *C, unsigned long long* errorsCount) {
    // printf("layer %d: [%dx%d] X [%dx%d] = [%dx%d]\n", layerIndex, M, K, K, N, M, N);
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    matrix_mult_dmr_kernel<THRESHOLD, CHECK_BLOCK><<<dimGrid,dimBlock>>>(A, B, M, N, K, C, errorsCount);
    check_error(cudaError_t(cudaPeekAtLastError()));
}

// > SHORTCUT

// E = A*B + C*D
__device__ __forceinline__ void e_axb_p_cxd(const float a, const float b, const float c, const float d, float &e) {
    e = __fma_rn(a, b, c*d);
}
__device__ __forceinline__ void e_axb_p_cxd(const float a, const float b, const float c, const float d, __half &e) {
    e = __hfma(__float2half(a), __float2half(b), __float2half(c)*__float2half(d));
}

template<const uint32_t THRESH, const uint32_t COUNT>
__global__ void shortcut_dmr_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out, unsigned long long *errorsCount) {
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    // out[out_index] = s1*out[out_index] + s2*add[add_index];

    register float out_f = 0.0;
	register __half out_h = 0.0;
    e_axb_p_cxd(s1, out[out_index], s2, add[add_index], out_f);
    e_axb_p_cxd(s1, out[out_index], s2, add[add_index], out_h);

    // if (COUNT == 1) {
        if (check_bit_error<THRESH>(out_h, out_f)) {
            atomicAdd(errorsCount, 1);
        }
    // }

    out[out_index] = out_f;
}

extern "C" void shortcut_dmr_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out, unsigned long long *errorsCount) {
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    int size = batch * minw * minh * minc;
    shortcut_dmr_kernel<THRESHOLD, CHECK_BLOCK><<<cuda_gridsize(size), BLOCK>>>(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, s1, s2, out, errorsCount);
    check_error(cudaPeekAtLastError());
}

// > ACTIVATION

template<typename T>
__device__ T lhtan_activate_kernel(T x)
{
    if(x < T(0)) return T(.001f)*x;
    if(x > T(1)) return T(.001f)*(x-T(1.f)) + T(1.f);
    return x;
}
template<typename T>
__device__ T lhtan_gradient_kernel(T x)
{
    if(x > T(0) && x < T(1)) return T(1);
    return T(.001);
}
template<typename T>
__device__ T hardtan_activate_kernel(T x)
{
    if (x < T(-1)) return T(-1);
    if (x > T(1)) return T(1);
    return x;
}
template<typename T>
__device__ T linear_activate_kernel(T x){return x;}

// __device__ float logistic_activate_kernel(float x){return 1.f/(1.f + expf(-x));}
__device__ __half logistic_activate_kernel(__half x){return __half(1.f)/(__half(1.f) + hexp(-x));}

// __device__ float loggy_activate_kernel(float x){return 2.f/(1.f + expf(-x)) - 1;}
__device__ __half loggy_activate_kernel(__half x){return __half(2.f)/(__half(1.f) + hexp(-x)) - __half(1);}

template<typename T>
__device__ T relu_activate_kernel(T x){return x*T(x>T(0));}

// __device__ float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(expf(x)-1);}
__device__ __half elu_activate_kernel(__half x){return __half(x >= __half(0))*x + __half(x < __half(0))*(hexp(x)-__half(1));}

// __device__ float selu_activate_kernel(float x){return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(expf(x)-1);}
__device__ __half selu_activate_kernel(__half x){return __half(x >= __half(0))*__half(1.0507f)*x + __half(x < __half(0))*__half(1.0507f)*__half(1.6732f)*(hexp(x)-__half(1));}

template<typename T>
__device__ T relie_activate_kernel(T x){return (x>T(0)) ? x : T(.01f)*x;}
template<typename T>
__device__ T ramp_activate_kernel(T x){return x*T(x>T(0))+T(.1f)*x;}
template<typename T>
__device__ T leaky_activate_kernel(T x){return (x>T(0)) ? x : T(.1f)*x;}

// __device__ float tanh_activate_kernel(float x){return (2.f/(1 + expf(-2*x)) - 1);}
__device__ __half tanh_activate_kernel(__half x){return (__half(2.f)/(__half(1) + hexp(__half(-2)*x)) - __half(1));}

template<typename T>
__device__ T plse_activate_kernel(T x)
{
    if(x < T(-4)) return T(.01f) * (x + T(4));
    if(x > T(4))  return T(.01f) * (x - T(4)) + T(1);
    return T(.125f)*x + T(.5f);
}
__device__ __half stair_activate_kernel(__half x)
{
    int n = hfloor(x);
    if (n%2 == 0) return hfloor(x/__half(2));
    else return (x - __half(n)) + hfloor(x/__half(2));
}

template<typename T>
__device__ T activate_dmr_kernel(T x, ACTIVATION a) {
    switch(a){
        case LINEAR:
            return linear_activate_kernel(x);
        case LOGISTIC:
            return logistic_activate_kernel(x);
        case LOGGY:
            return loggy_activate_kernel(x);
        case RELU:
            return relu_activate_kernel(x);
        case ELU:
            return elu_activate_kernel(x);
        case SELU:
            return selu_activate_kernel(x);
        case RELIE:
            return relie_activate_kernel(x);
        case RAMP:
            return ramp_activate_kernel(x);
        case LEAKY:
            return leaky_activate_kernel(x);
        case TANH:
            return tanh_activate_kernel(x);
        case PLSE:
            return plse_activate_kernel(x);
        case STAIR:
            return stair_activate_kernel(x);
        case HARDTAN:
            return hardtan_activate_kernel(x);
        case LHTAN:
            return lhtan_activate_kernel(x);
    }
    return 0;
}

template<const uint32_t THRESH, const uint32_t COUNT>
__global__ void activate_array_dmr_kernel(float *x, int n, ACTIVATION a, unsigned long long *errorsCount) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
        // x[i] = activate_dmr_kernel(x[i], a);

        register float out_f = activate_dmr_kernel<float>(x[i], a);
        register __half out_h = activate_dmr_kernel<__half>(__float2half(x[i]), a);

        if (COUNT == 1) {
            if (check_bit_error<THRESH>(out_h, out_f)) {
                atomicAdd(errorsCount, 1);
            }
        }

        x[i] = out_f;
    }
}

extern "C" void activate_array_dmr_gpu(float *x, int n, ACTIVATION a, unsigned long long *errorsCount) {
    activate_array_dmr_kernel<THRESHOLD, CHECK_BLOCK><<<cuda_gridsize(n), BLOCK>>>(x, n, a, errorsCount);
    check_error(cudaPeekAtLastError());
}

// > UPSAMPLE

template<const uint32_t THRESH, const uint32_t COUNT>
__global__ void upsample_dmr_kernel(size_t N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out, unsigned long long *errorsCount) {
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;

    register float out_float = out[out_index];
    register __half out_half = __float2half(out[out_index]);

    axpy__(scale, x[in_index], out_float);
    axpy__(scale, x[in_index], out_half);

    if (COUNT == 1) {
        if (check_bit_error<THRESH>(out_half, out_float)) {
            atomicAdd(errorsCount, 1);
        }
    }

    // if(forward) out[out_index] += scale * x[in_index];
    // else atomicAdd(x+in_index, scale * out[out_index]);
    
    if(forward) out[out_index] = out_float;
    else atomicAdd(x+in_index, scale * out[out_index]);
}

extern "C" void upsample_dmr_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out, unsigned long long *errorsCount) {
    size_t size = w*h*c*batch*stride*stride;
    upsample_dmr_kernel<THRESHOLD, CHECK_BLOCK><<<cuda_gridsize(size), BLOCK>>>(size, in, w, h, c, batch, stride, forward, scale, out, errorsCount);
    check_error(cudaPeekAtLastError());
}

#endif /* DMR_KERNELS_H */