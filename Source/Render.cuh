#ifndef RENDER
#define RENDER

#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#include "Hit/HittableObjectsDevice.cuh"

struct CUDA_Render
{
    uint32_t* gpuFramebuffer;
    curandState *randomState;
};

CUDA_Render CUDA_render_setup(int width, int height, int pixelSamples);
void CUDA_render_destroy(CUDA_Render* cudaRender);
// Section
// 0 = Top left
// 1 = Top right
// 2 = Bottom left
// 3 = Bottom right
__global__ void CUDA_render_render_(uint32_t* framebuffer, curandState* randomState, unsigned short section, int windowWidth, int windowHeight, int pixelSamples, Camera camera, const HittableObjectsDevice world, Surface& s);

#endif
