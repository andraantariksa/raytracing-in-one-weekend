#include <cstdint>
#include <iostream>

#include "surface.hpp"
#include "color.hpp"

Surface::Surface(uint32_t* framebuffer, int width, int m_height):
	m_framebuffer(framebuffer),
	m_width(width),
	m_height(m_height)
{
}

void Surface::setPixel(int x, int y, Color& color)
{
	int col = 0x00000000;
	//std::cout << static_cast<int>(255.0f * color.r) << '\n';
	col |= static_cast<int>(255.0f * color.r) << 16;
	col |= static_cast<int>(255.0f * color.g) << 8;
	col |= static_cast<int>(255.0f * color.b);

	m_framebuffer[m_width * y + x] = col;
}
