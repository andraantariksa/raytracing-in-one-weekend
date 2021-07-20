#include <cstdint>
#include <iostream>
#include <algorithm>

#include "Surface.hpp"
#include "typedef.hpp"

Surface::Surface(SDL_Window* window, int width, int m_height):
    m_window(window),
    m_width(width),
    m_height(m_height),
    m_surface(SDL_GetWindowSurface(window)),
    m_frameBuffer((uint32_t*)m_surface->pixels)
{
}

void Surface::setPixel(int x, int y, Color& color, int samples)
{
    Color colorScaled = color / (float)samples;

	int col = 0x00000000;
	//std::cout << static_cast<int>(255.0f * color.r) << '\n';
	col |= std::clamp(static_cast<int>(255.0f * colorScaled.r), 0, 255) << 16;
	col |= std::clamp(static_cast<int>(255.0f * colorScaled.g), 0, 255) << 8;
	col |= std::clamp(static_cast<int>(255.0f * colorScaled.b), 0, 255);

    m_frameBuffer[m_width * y + x] = col;
}

void Surface::setDrawFunc(const std::function<void(Surface&)>& func)
{
    m_drawFunc = func;
}

void Surface::draw()
{
    SDL_LockSurface(m_surface);

    m_drawFunc(*this);

    SDL_UnlockSurface(m_surface);
    SDL_UpdateWindowSurface(m_window);
}
