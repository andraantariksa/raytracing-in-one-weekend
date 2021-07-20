#include <cstdint>
#include "typedef.hpp"
#include "Camera.hpp"
#include "Hit/HittableObjects.hpp"
#include <random>
//#include "Misc/Random.hpp"
#include "Surface.hpp"

uint32_t* gpuFramebuffer;

void CUDA_render_setup(int width, int height)
{
    cudaMallocManaged(&gpuFramebuffer, sizeof(uint32_t) * width * height);
}

void CUDA_render_destroy()
{
    cudaFree(gpuFramebuffer);
}

__global__ void CUDA_render_render(int windowWidth, int windowHeight, int pixelSamples, Camera camera, const HittableObjects& world, Surface& s)
{
    for (int i = 0; i < windowHeight; i++)
    {
        for (int j = 0; j < windowWidth; j++)
        {
            Color accColor(0.0f);
            for (int s = 0; s < pixelSamples; s++)
            {
//                float u = (float)(j + randomIntOne()) / (float)windowWidth;
//                float v = (float)(i + randomIntOne()) / (float)windowHeight;
//                accColor += rayColor(camera.getRay(u, v), world);
            }
//            s.setPixel(j, i, accColor, pixelSamples);
        }
    }
}
