#include <cstdint>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>
#include <algorithm>
#include <iostream>

#include "typedef.cuh"
#include "Camera.cuh"
#include "Hit/HittableObjects.cuh"
#include "Hit/HittableObjectsDevice.cuh"
#include "Misc/Random.cuh"
#include "Surface.cuh"
#include "Render.cuh"

CUDA_Render CUDA_render_setup(int width, int height, int pixelSamples)
{
    CUDA_Render cudaRender{};
    cudaMalloc((void**)&cudaRender.gpuFramebuffer, sizeof(uint32_t) * width * height);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc((void**)&cudaRender.randomState, width * height * pixelSamples *sizeof(curandState));
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return cudaRender;
}

void CUDA_render_destroy(CUDA_Render* cudaRender)
{
    cudaFree(cudaRender->gpuFramebuffer);
    cudaFree(cudaRender->randomState);
}

__device__ Color CUDA_rayColor(const Ray& ray, const HittableObjectsDevice& world)
{
    HitData hitData;
    bool hasValue = false;
    world.hit(hitData, hasValue, ray, 0.0f, std::numeric_limits<float>::infinity());
    if (hasValue)
    {
        return 0.5f * Color(std::max(hitData.N.x + 1.0f, 1.0f),
            std::max(hitData.N.y + 1.0f, 1.0f),
            std::max(hitData.N.z + 1.0f, 1.0f));
    }

    auto directionNorm = glm::normalize(ray.direction());
    auto t = (directionNorm.y + 1.0f) * 0.5f;
    return glm::lerp(Color(0.67f, 0.84f, 0.92f), Color(1.0f), glm::vec3(t));
}

__global__ void CUDA_render_init(curandState* randomState, unsigned short section, int windowWidth, int windowHeight)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    int framebufferIdx = windowWidth * j + i;

    curand_init(0, framebufferIdx, 0, &randomState[framebufferIdx]);
}

__global__ void CUDA_render_render_(uint32_t* framebuffer, curandState* randomState, unsigned short section, int windowWidth, int windowHeight, int pixelSamples, Camera camera, const HittableObjectsDevice world, std::time_t time)
{
    // DO NOT modify threadIdx.x or blockIdx.x by adding them directly, copy their value first!
    int i = blockIdx.x;
    int j = threadIdx.x;
    glm::vec2 coord(i, j);

    int framebufferIdx = windowWidth * j + i;

    Color accColor(0.0f);

    auto localRandomState = randomState[framebufferIdx];

    for (int s = 0; s < pixelSamples; s++)
    {
//        float u = ((float)i + goldenNoise(coord, time + s)) / (float)windowWidth;
        float u = ((float)i + curand_uniform(&localRandomState)) / (float)windowWidth;
//        float v = ((float)j + goldenNoise(coord, time + s)) / (float)windowHeight;
        float v = ((float)j + curand_uniform(&localRandomState)) / (float)windowHeight;
        accColor += CUDA_rayColor(camera.getRay(u, v), world);
    }
    Color colorScaled = accColor / (float)pixelSamples;

    int col = 0x00000000;
    col |= std::clamp(static_cast<int>(255.0f * colorScaled.r), 0, 255) << 16;
    col |= std::clamp(static_cast<int>(255.0f * colorScaled.g), 0, 255) << 8;
    col |= std::clamp(static_cast<int>(255.0f * colorScaled.b), 0, 255);

    framebuffer[framebufferIdx] = col;
}
