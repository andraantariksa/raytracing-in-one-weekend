#include <cstdint>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>
#include <algorithm>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <iostream>

#include "Typedef.cuh"
#include "Camera.cuh"
#include "Hit/HittableObjectsDevice.cuh"
#include "Util/Random.cuh"
#include "Surface.cuh"
#include "Render.cuh"

__global__ void CUDA_render_init(CUDARenderer renderer)
{
    auto framebufferIdx = renderer.framebufferIndex(blockIdx.x, threadIdx.x);

    curand_init(0, framebufferIdx, 0, &thrust::raw_pointer_cast(renderer.getRandomStates())[framebufferIdx]);
}

__global__ void CUDA_render_render(CUDARenderer renderer)
{
    // DO NOT modify threadIdx.x or blockIdx.x by adding them directly, copy their value first!
    auto i = blockIdx.x;
    auto j = threadIdx.x;
    auto framebufferIdx = renderer.framebufferIndexFlipped(i, j);
    auto localRandomState = thrust::raw_pointer_cast(renderer.getRandomStates())[framebufferIdx];
    Color accColor(0.0f);

    for (int s = 0; s < renderer.getAntiAliasingPixelSamples(); s++)
    {
        float u = ((float)i + curand_uniform(&localRandomState)) / (float)renderer.getWidth();
        float v = ((float)j + curand_uniform(&localRandomState)) / (float)renderer.getHeight();
        accColor += renderer.rayColor(renderer.getCamera().getRay(u, v), renderer.getWorld(), &localRandomState, framebufferIdx, renderer.getMaxLightBounce());
    }
    float colorScale = 1.0f / (float)renderer.getAntiAliasingPixelSamples();
    float gammaPower = 1.0f / renderer.getGamma();
    glm::vec3 colorScaled = glm::pow(accColor * colorScale, glm::vec3(gammaPower));

    int col = 0x00000000;
    col |= std::clamp(static_cast<int>(255.0f * colorScaled.r), 0, 255) << 16;
    col |= std::clamp(static_cast<int>(255.0f * colorScaled.g), 0, 255) << 8;
    col |= std::clamp(static_cast<int>(255.0f * colorScaled.b), 0, 255);

    thrust::raw_pointer_cast(renderer.getGPUFramebuffer())[framebufferIdx] = col;
}
