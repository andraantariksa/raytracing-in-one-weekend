#ifndef RAYTRACING_SOURCE_SURFACE_CUH
#define RAYTRACING_SOURCE_SURFACE_CUH

#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cstdint>
#include <SDL2/SDL.h>
#include "Typedef.cuh"
#include <functional>

class Surface
{
public:
	Surface(SDL_Window* window, int width, int height);
    void setDrawFunc(const std::function<void(Surface&)>& func);
    void draw();
    void setPixel(int x, int y, Color& color, int samples = 1);
    void copyFramebufferHostToDevice(uint32_t* deviceBuffer);
    void copyFramebufferDeviceToHost(uint32_t* deviceBuffer);
private:
    std::function<void(Surface&)> m_drawFunc;
    SDL_Window* m_window;
	SDL_Surface* m_surface;
	int m_width;
	int m_height;
    uint32_t* m_frameBuffer;
};



#endif
