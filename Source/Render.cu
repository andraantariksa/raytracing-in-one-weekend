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

#include "typedef.cuh"
#include "Camera.cuh"
#include "Hit/HittableObjects.cuh"
#include "Hit/HittableObjectsDevice.cuh"
#include "Misc/Random.cuh"
#include "Surface.cuh"
#include "Render.cuh"

class CUDARenderer
{
public:
    CUDARenderer(int width, int height, int pixelSamples = 100, int maxRayBounce = 50, float gamma = 2.0f):
        m_gpuFramebuffer(thrust::device_new<uint32_t>(width * height)),
        m_renderStates(thrust::device_new<curandState>(width * height))
    {
    }

    __device__ void render()
    {

    }

    ~CUDARenderer()
    {
        thrust::device_delete(m_gpuFramebuffer);
        thrust::device_delete(m_renderStates);
    }
private:
    thrust::device_ptr<uint32_t> m_gpuFramebuffer;
    thrust::device_ptr<curandState> m_renderStates;
};

CUDA_Render CUDA_render_setup(int width, int height, int pixelSamples)
{
    CUDA_Render cudaRender{};
    cudaMalloc((void**)&cudaRender.gpuFramebuffer, sizeof(uint32_t) * width * height);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc((void**)&cudaRender.randomState, width * height * pixelSamples * sizeof(curandState));
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return cudaRender;
}

void CUDA_render_destroy(CUDA_Render* cudaRender)
{
    cudaFree(cudaRender->gpuFramebuffer);
    cudaFree(cudaRender->randomState);
}

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
        bool hasValue = false;
        world.hit(hitData, hasValue, currentRay, 0.001f, std::numeric_limits<float>::infinity());
        if (hasValue)
        {
            auto target = hitData.coord + hitData.N + glm::normalize(randomInHemisphere(hitData.N, localRandomState, framebufferIdx));
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

__global__ void CUDA_render_init(curandState* randomState, unsigned short section, int windowWidth, int windowHeight)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    int framebufferIdx = windowWidth * j + i;

    curand_init(0, framebufferIdx, 0, &randomState[framebufferIdx]);
}

__global__ void CUDA_render_render_(uint32_t* framebuffer,
    curandState* randomState,
    unsigned short section,
    int windowWidth,
    int windowHeight,
    int pixelSamples,
    Camera camera,
    const HittableObjectsDevice world,
    std::time_t time,
    int maxRecursionDepth)
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
        float u = ((float)i + curand_uniform(&localRandomState)) / (float)windowWidth;
        float v = ((float)j + curand_uniform(&localRandomState)) / (float)windowHeight;
        accColor += CUDA_rayColor(camera.getRay(u, v), world, &localRandomState, framebufferIdx, maxRecursionDepth);
    }
    float gamma = 2.0f;
    float colorScale = 1.0f / (float)pixelSamples;
    float gammaPower = 1.0f / gamma;
    glm::vec3 colorScaled = glm::pow(accColor * colorScale, glm::vec3(gammaPower));

    int col = 0x00000000;
    col |= std::clamp(static_cast<int>(255.0f * colorScaled.r), 0, 255) << 16;
    col |= std::clamp(static_cast<int>(255.0f * colorScaled.g), 0, 255) << 8;
    col |= std::clamp(static_cast<int>(255.0f * colorScaled.b), 0, 255);

    framebuffer[framebufferIdx] = col;
}
