#ifndef DMR_TYPES_H
#define DMR_TYPES_H

typedef struct {
    unsigned long long errors;
} LayerErrors;

#define LAYERS 107
// __device__ LayerErrors errorsPerLayer[LAYERS];
// __device__ int errs = 0;

#endif