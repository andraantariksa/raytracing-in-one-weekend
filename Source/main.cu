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
#include "Hit/Object/SphereObject.cuh"
#include "Camera.cuh"
#include "Render.cuh"
#include "Util/Random.cuh"
#include "Hit/HittableObjectsDevice.cuh"
#include "Util/SDLHelpers.cuh"
#include "Util/Timer.cuh"
#include "Material/Metal.cuh"
#include "Material/Lambertian.cuh"

__global__ void init(IHittableObject** objects)
{
    objects[0] = new SphereObject(glm::vec3(-4.0f, 0.0f, -10.0f), 4.0f, new Metal(Color(0.753f)));
    objects[1] = new SphereObject(glm::vec3(4.0f, 0.0f, -10.0f), 4.0f, new Lambertian(Color(0.0f, 0.753f, 0.0f)));
    objects[2] = new SphereObject(glm::vec3(0.0f, -104.0f, -10.0f), 100.0f, new Lambertian(Color(1.0f, 0.0f, 0.0f)));
}

int main()
{
    assert(SDL_Init(SDL_INIT_VIDEO) == 0);

    const float aspectRatio = 16.0f / 9.0f;
    const float windowHeight = 720;
    const float windowWidth = windowHeight * aspectRatio;
    const float viewportWidth = 10.0f * ((float)windowWidth / (float)windowHeight);
    const float viewportHeight = 10.0f;
    const float vocalLength = 10.0f;
    int pixelSamples = 100;
    int maxRecursionDepth = 50;

    auto* window =
        SDL_CreateWindow("Raytracing", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, windowWidth, windowHeight, 0);
    assert(window != nullptr);

    // Draw
    Surface surf(window, windowWidth, windowHeight);
    Camera camera(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f), M_PI / 2.0f, aspectRatio);

    HittableObjectsDevice worldDevice(3);
    init<<<1, 1>>>(worldDevice.getObjects());

    CUDARenderer renderer(windowWidth, windowHeight, camera, worldDevice, pixelSamples, maxRecursionDepth, 2.0f);

    surf.setDrawFunc([&](auto s)
    {
        renderer.render();
        surf.copyFramebufferDeviceToHost(thrust::raw_pointer_cast(renderer.getGPUFramebuffer()));
    });
    surf.draw();

    // End draw

    Timer update;
    Timer fps;

    update.start();
    fps.start();
    int frame = 0;

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
            {
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
