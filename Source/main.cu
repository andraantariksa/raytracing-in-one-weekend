﻿#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <cassert>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/common.hpp>
#include <utility>
#include <iostream>

#include "typedef.cuh"
#include "Surface.cuh"
#include "Ray.cuh"
#include "Hit/HittableObjects.cuh"
#include "Camera.cuh"
#include "Hit/Object/SphereObject.cuh"
#include "Render.cuh"
#include "Misc/Random.cuh"
#include <ppl.h>

Color rayColor(const Ray& ray, const HittableObjects& world)
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

__global__ void r(uint32_t * cudaRender, int windowWidth, int windowHeight, int pixelSamples, Camera camera, const HittableObjects& world, Surface& s)
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    cudaRender[windowWidth * i + j] = 0x000000FF;
}

int main()
{
    assert(SDL_Init(SDL_INIT_VIDEO) == 0);
    const int windowWidth = 1280;
    const int windowHeight = 720;
    const float viewportWidth = 10.0f * ((float)windowWidth / (float)windowHeight);
    const float viewportHeight = 10.0f;
    const float vocalLength = 10.0f;
    int pixelSamples = 5;
    auto* window =
        SDL_CreateWindow("Raytracing", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, windowWidth, windowHeight, 0);
    assert(window != nullptr);

//    CUDA_Render cudaRender = CUDA_render_setup(windowWidth, windowHeight, pixelSamples);
    CUDA_Render cudaRender{};
    cudaMalloc((void**)&cudaRender.gpuFramebuffer, sizeof(uint32_t) * windowWidth * windowHeight);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc((void**)&cudaRender.randomState, windowWidth * windowHeight * pixelSamples *sizeof(curandState));
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    // Draw
    Surface surf(window, windowWidth, windowHeight);
    Camera camera(glm::vec3(0.0f), viewportWidth, viewportHeight, vocalLength);

    HittableObjects world;
    world.add(std::make_shared<SphereObject>(SphereObject(glm::vec3(0.0f, 0.0f, -10.0f), 3.0f)));
    world.add(std::make_shared<SphereObject>(SphereObject(glm::vec3(-4.0f, 0.0f, -10.0f), 4.0f)));

    surf.setDrawFunc([&](auto s)
    {
//        std::cout << "Draw\n";
        surf.copyFramebufferHostToDevice(cudaRender.gpuFramebuffer);
        // The rendering process is splitted into 4 region. See Render.cuh
        for (unsigned short section = 0; section < 4; section++)
        {
            CUDA_render_render_<<<640, 360>>>(cudaRender.gpuFramebuffer,
                section,
                windowWidth,
                windowHeight,
                pixelSamples,
                camera,
                world,
                s);
        }
//        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
        surf.copyFramebufferDeviceToHost(cudaRender.gpuFramebuffer);
//        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    });
    surf.draw();

    // End draw

    SDL_Event event;

    bool running = true;
    bool isNeedToRedraw = false;
    while (running)
    {
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym)
                {
                case SDLK_a:
                    camera.transform(glm::translate(glm::mat4(1.0f), glm::vec3(-1.0f, 0.0f, 0.0f)));
                    break;
                case SDLK_d:
                    camera.transform(glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 0.0f)));
                    break;
                case SDLK_w:
                    camera.transform(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.0f, 0.0f)));
                    break;
                case SDLK_s:
                    camera.transform(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.0f, 0.0f)));
                    break;
                default:
                    break;
                }
                isNeedToRedraw = true;
                break;
            }
        }

//        if (isNeedToRedraw)
//        {
            surf.draw();
//            isNeedToRedraw = false;
//        }
    }

    CUDA_render_destroy(&cudaRender);
    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}