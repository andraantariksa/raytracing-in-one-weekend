#include <SDL2/SDL.h>
#include <cassert>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/compatibility.hpp>
#include <thrust/device_malloc.h>
#include <glm/common.hpp>
#include <utility>
#include <iostream>
#include <ctime>
#include <sstream>
#include <ppl.h>

#include "Typedef.cuh"
#include "Surface.cuh"
#include "Ray.cuh"
#include "Hit/HittableObjects.cuh"
#include "Hit/Object/SphereObject.cuh"
#include "Camera.cuh"
#include "Render.cuh"
#include "Util/Random.cuh"
#include "Hit/HittableObjectsDevice.cuh"
#include "Util/SDLHelpers.cuh"
#include "Util/Timer.cuh"

int main()
{
    assert(SDL_Init(SDL_INIT_VIDEO) == 0);

    const int windowWidth = 1280;
    const int windowHeight = 720;
    const float viewportWidth = 10.0f * ((float)windowWidth / (float)windowHeight);
    const float viewportHeight = 10.0f;
    const float vocalLength = 10.0f;
    int pixelSamples = 100;
    int maxRecursionDepth = 50;

    auto* window =
        SDL_CreateWindow("Raytracing", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, windowWidth, windowHeight, 0);
    assert(window != nullptr);

//    CUDA_Render cudaRender{};
//    cudaMalloc((void**)&cudaRender.gpuFramebuffer, sizeof(uint32_t) * windowWidth * windowHeight);
//    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
//    cudaMalloc((void**)&cudaRender.randomState, sizeof(curandState) * windowWidth * windowHeight);
//    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    // Draw
    Surface surf(window, windowWidth, windowHeight);
    Camera camera(glm::vec3(0.0f), viewportWidth, viewportHeight, vocalLength);

//    HittableObjects world;
//    world.add(std::make_shared<SphereObject>(SphereObject(glm::vec3(0.0f, 0.0f, -10.0f), 3.0f)));
//    world.add(std::make_shared<SphereObject>(SphereObject(glm::vec3(-4.0f, 0.0f, -10.0f), 4.0f)));

    HittableObjectsDevice worldDevice(2);

    SphereObject* od_1;
    cudaMalloc(&od_1, sizeof(SphereObject));
    SphereObject oh_1(glm::vec3(0.0f, 0.0f, -10.0f), 3.0f);
    cudaMemcpy(od_1, &oh_1, sizeof(SphereObject), cudaMemcpyHostToDevice);
    cudaMemcpy(&worldDevice.m_objects[0], &od_1, sizeof(IHittableObject*), cudaMemcpyHostToDevice);
//    worldDevice.set(0, (IHittableObject **)&od_1);

    SphereObject* od_2;
    cudaMalloc(&od_2, sizeof(SphereObject));
    SphereObject oh_2(glm::vec3(-4.0f, 0.0f, -10.0f), 4.0f);
    cudaMemcpy(od_2, &oh_2, sizeof(SphereObject), cudaMemcpyHostToDevice);
    cudaMemcpy(&worldDevice.m_objects[1], &od_2, sizeof(IHittableObject*), cudaMemcpyHostToDevice);
//    worldDevice.set(1, (IHittableObject **)&od_2);

    CUDARenderer renderer(windowWidth, windowHeight, camera, worldDevice, pixelSamples, maxRecursionDepth, 2.0f);

//    CUDA_render_init_<<<1280, 720>>>(
//        cudaRender.randomState,
//        windowWidth,
//        windowHeight);
//    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    surf.setDrawFunc([&](auto s)
    {
//      CUDA_render_render_<<<1280, 720>>>(cudaRender.gpuFramebuffer,
//          cudaRender.randomState,
//          windowWidth,
//          windowHeight,
//          pixelSamples,
//          camera,
//          worldDevice,
//          maxRecursionDepth);
//        surf.copyFramebufferDeviceToHost(cudaRender.gpuFramebuffer);
        renderer.render();
        surf.copyFramebufferDeviceToHost(thrust::raw_pointer_cast(renderer.getGPUFramebuffer()));
    });
    surf.draw();

    // End draw

    Timer update;
    update.start();
    Timer fps;
    fps.start();

    SDL_Event event;
    int frame = 0;

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
            {
                switch (event.key.keysym.sym)
                {
                case SDLK_a:
                    std::cout << "A pressed\n";
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
        }

        if (isNeedToRedraw)
        {
            surf.draw();
            isNeedToRedraw = false;
        }

        frame++;
        if(update.get_ticks() > 1000)
        {
            std::stringstream caption;
            caption << "Average Frames Per Second: " << frame / ( fps.get_ticks() / 1000.f );
            SDL_SetWindowTitle(window, caption.str().c_str());
            update.start();
        }
    }

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
