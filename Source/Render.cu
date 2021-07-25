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

__global__ void CUDA_render_init(CUDARenderer renderer);

__global__ void CUDA_render_render(CUDARenderer renderer);

__device__ Color CUDA_rayColor(const Ray& ray,
    const HittableObjectsDevice& world,
    curandState* localRandomState,
    int framebufferIdx,
    int maxRecursionDepth)
{
    float currentAttenuation = 1.0f;
    Ray currentRay = ray;
    for (int i = 0; i < maxRecursionDepth; ++i)
    {
        HitData hitData;
        if (world.hit(hitData, currentRay, 0.001f, std::numeric_limits<float>::infinity()))
        {
            auto target = hitData.coord + hitData.N
                + glm::normalize(randomInHemisphere(hitData.N, localRandomState, framebufferIdx));
            currentAttenuation *= 0.5f;
            currentRay = Ray(hitData.coord, target - hitData.coord);
        }
        else
        {
            auto directionNorm = glm::normalize(currentRay.direction());
            auto t = (directionNorm.y + 1.0f) * 0.5f;
            auto color = glm::lerp(Color(0.67f, 0.84f, 0.92f), Color(1.0f), glm::vec3(t));
            return currentAttenuation * color;
        }
    }

    return Color(0.0f, 0.0f, 0.0f);
}

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
        accColor += CUDA_rayColor(renderer.getCamera().getRay(u, v), renderer.getWorld(), &localRandomState, framebufferIdx, renderer.getMaxLightBounce());
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
