#ifndef RENDER
#define RENDER

#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <iostream>

#include "Hit/HittableObjectsDevice.cuh"
#include "Typedef.cuh"
#include "Camera.cuh"
#include "Hit/HittableObjectsDevice.cuh"
#include "Util/Random.cuh"
#include "Surface.cuh"
#include "Render.cuh"

class CUDARenderer;

__global__ void CUDA_render_init(CUDARenderer renderer);
__global__ void CUDA_render_render(CUDARenderer renderer);

class CUDARenderer
{
public:
    CUDARenderer(int width,
        int height,
        Camera& camera,
        HittableObjectsDevice& world,
        int antiAliasingPixelSamples = 100,
        int maxRayBounce = 50,
        float gamma = 2.0f) :
        m_gpuFramebuffer(thrust::device_new<uint32_t>(width * height)),
        m_randomStates(thrust::device_new<curandState>(width * height)),
        m_width(width),
        m_height(height),
        m_camera(camera),
        m_world(world),
        m_maxLightBounce(maxRayBounce),
        m_antiAliasingPixelSamples(antiAliasingPixelSamples),
        m_gamma(gamma)
    {
        CUDA_render_init<<<m_width, m_height>>>(*this);
    }

    __device__ Color rayColor(const Ray& ray,
        const HittableObjectsDevice& world,
        curandState* localRandomState,
        int framebufferIdx,
        int maxRecursionDepth)
    {
//        float currentAttenuation = 1.0f;
//        Ray currentRay = ray;
//        for (int i = 0; i < maxRecursionDepth; ++i)
//        {
//            HitData hitData;
//            bool hasValue = false;
//            world.hit(hitData, hasValue, currentRay, 0.001f, std::numeric_limits<float>::infinity());
//            if (hasValue)
//            {
//                auto target = hitData.coord + hitData.N
//                    + glm::normalize(randomInHemisphere(hitData.N, localRandomState, framebufferIdx));
//                currentAttenuation *= 0.5f;
//                currentRay = Ray(hitData.coord, target - hitData.coord);
//            }
//            else
//            {
//                auto directionNorm = glm::normalize(currentRay.direction());
//                auto t = (directionNorm.y + 1.0f) * 0.5f;
//                auto color = glm::lerp(Color(0.67f, 0.84f, 0.92f), Color(1.0f), glm::vec3(t));
//                return currentAttenuation * color;
//            }
//        }

        return Color(1.0f, 0.0f, 0.0f);
    }

    __device__ __host__ inline uint32_t framebufferIndexFlipped(uint32_t x, uint32_t y)
    {
        return m_width * (m_height - 1 - y) + x;
    }

    __device__ __host__ inline uint32_t framebufferIndex(uint32_t x, uint32_t y)
    {
        return m_width * y + x;
    }

    void render()
    {
        CUDA_render_render<<<m_width, m_height>>>(*this);
    }

    void destroy()
    {
        thrust::device_delete(m_gpuFramebuffer);
        thrust::device_delete(m_randomStates);
    }

    __device__ __host__ inline const thrust::device_ptr<uint32_t>& getGPUFramebuffer() const
    {
        return m_gpuFramebuffer;
    }

    __device__ __host__ inline const thrust::device_ptr<curandState>& getRandomStates() const
    {
        return m_randomStates;
    }

    __device__ __host__ inline int getWidth() const
    {
        return m_width;
    }

    __device__ __host__ inline int getHeight() const
    {
        return m_height;
    }

    __device__ __host__ inline const Camera& getCamera() const
    {
        return m_camera;
    }

    __device__ __host__ inline const HittableObjectsDevice& getWorld() const
    {
        return m_world;
    }

    __device__ __host__ inline int getMaxLightBounce() const
    {
        return m_maxLightBounce;
    }

    __device__ __host__ inline int getAntiAliasingPixelSamples() const
    {
        return m_antiAliasingPixelSamples;
    }

    __device__ __host__ inline float getGamma() const
    {
        return m_gamma;
    }

private:
    thrust::device_ptr<uint32_t> m_gpuFramebuffer;
    thrust::device_ptr<curandState> m_randomStates;
    int m_width;
    int m_height;
    Camera m_camera;
    HittableObjectsDevice m_world;
    int m_maxLightBounce;
    int m_antiAliasingPixelSamples;
    float m_gamma;
};

#endif
