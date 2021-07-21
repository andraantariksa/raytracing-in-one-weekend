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
//#include "Misc/Random.hpp"
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

__device__ Color CUDA_rayColor(const Ray& ray, const HittableObjects& world)
{
    auto hit = world.hit(ray, 0.0f, std::numeric_limits<float>::infinity());
    if (hit.has_value())
    {
        auto hitData = hit.value();
        return 0.5f * Color(std::max(hitData.N.x + 1.0f, 1.0f),
            std::max(hitData.N.y + 1.0f, 1.0f),
            std::max(hitData.N.z + 1.0f, 1.0f));
    }

    auto directionNorm = glm::normalize(ray.direction());
    auto t = (directionNorm.y + 1.0f) * 0.5f;
    return glm::lerp(Color(0.67f, 0.84f, 0.92f), Color(1.0f), glm::vec3(t));
}

__global__ void CUDA_render_render_(uint32_t* framebuffer, unsigned short section, int windowWidth, int windowHeight, int pixelSamples, Camera camera, const HittableObjects& world, Surface& s)
{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    curand_init(0, idx, 0, &cudaRender->randomState[idx]);
//    float random = curand_uniform(&cudaRender->randomState[idx]);
    // DO NOT modify threadIdx.x or blockIdx.x by adding them directly, copy their value first!
    int i = threadIdx.x;
    i += (section > 1) ? windowHeight / 2 : 0;
    int j = blockIdx.x;
    j += (section % 2 != 0) ? windowWidth / 2 : 0;
    Color accColor(0.0f);
    for (int s = 0; s < pixelSamples; s++)
    {
        float u = (float)(j) / (float)windowWidth;
        float v = (float)(i) / (float)windowHeight;
//                accColor += CUDA_rayColor(camera.getRay(u, v), world);
    }
//            s.setPixel(j, i, accColor, pixelSamples);
//            Color colorScaled = accColor / (float)pixelSamples;

//    int col = 0x000000FF;
//            col |= std::clamp(static_cast<int>(255.0f * colorScaled.r), 0, 255) << 16;
//            col |= std::clamp(static_cast<int>(255.0f * colorScaled.g), 0, 255) << 8;
//            col |= std::clamp(static_cast<int>(255.0f * colorScaled.b), 0, 255);

//    framebuffer[windowWidth * i + j] = col;
}
