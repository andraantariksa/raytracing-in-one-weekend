#ifndef _RT_SURFACE_HPP
#define _RT_SURFACE_HPP

#include <cstdint>

#include "color.hpp"

class Surface
{
public:
	Surface(uint32_t* framebuffer, int width, int m_height);
	void setPixel(int x, int y, Color& color);
private:
	uint32_t* m_framebuffer;
	int m_width;
	int m_height;
};



#endif
