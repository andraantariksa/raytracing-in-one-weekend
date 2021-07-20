#ifndef _RT_SURFACE_HPP
#define _RT_SURFACE_HPP

#include <cstdint>
#include <SDL2/SDL.h>
#include "typedef.hpp"
#include <functional>

class Surface
{
public:
	Surface(SDL_Window* window, int width, int m_height);
	void setDrawFunc(const std::function<void(Surface&)>& func);
	void draw();
	void setPixel(int x, int y, Color& color, int samples = 1);
private:
    std::function<void(Surface&)> m_drawFunc;
    SDL_Window* m_window;
	SDL_Surface* m_surface;
	uint32_t* m_frameBuffer;
	int m_width;
	int m_height;
};



#endif
