#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <cassert>
#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/common.hpp>
#include "color.hpp"
#include "surface.hpp"
#include "ray.hpp"

bool hitSphere(const glm::vec3& center, const float r, const Ray& ray)
{
	// P = Point lies in the sphere surface (Vector)
	// C = center of sphere (Vector)
	// (x - C_x)^2 + (y - C_y)^2 + (z + C_z)^2 = r^2
	// To vector notation
	// (P - C) ⋅ (P - C) = r^2
	// 
	// b = Direction of ray (Vector)
	// A = Ray origin or camera (Vector)
	// t = Scale of direction to touch the object that are hitable by  (Scalar)ray
	// P(t) = A + t * b
	//
	// (P(t) - C) ⋅ (P(t) - C) = r^2
	// (A + t * b - C) ⋅ (A + t * b - C) = r^2
	// (t * b + (A - C)) ⋅ (t * b + (A - C)) = r^2
	// (tb)^2 + t * b ⋅ (A - C) * 2 + (A - C) ⋅ (A - C) = r^2
	// t^2 * b ⋅ b + t * b ⋅ (A - C) * 2 + (A - C) ⋅ (A - C) - r^2 = 0
	// Every variable is known except t. Turn it into quadratic
	// (b ⋅ b) * t^2 + 2 * b ⋅ (A - C) * t + (A - C) ⋅ (A - C) - r^2 = 0
	//
	// ax^2 + bx + c
	// a = b ⋅ b
	// b = 2 * b ⋅ (A - C)
	// c = (A - C) ⋅ (A - C) - r^2
	auto ASubC = ray.origin() - center;
	auto a = glm::dot(ray.direction(), ray.direction());
	auto b = 2.0f * glm::dot(ray.direction(), ASubC);
	auto c = glm::dot(ASubC, ASubC) - r * r;
	auto discriminant = b * b - 4 * a * c;
	return discriminant >= 0.0f;
}

Color rayColor(const Ray& ray)
{
	if (hitSphere(glm::vec3(0.0f, 0.0f, -10.0f), 3.0f, ray))
	{
		return Color(1.0f, 0.0f, 0.0f);
	}

	auto directionNorm = glm::normalize(ray.direction());
	auto t = (directionNorm.y + 1.0f) * 0.5f;
	return glm::lerp(Color(0.67f, 0.84f, 0.92f), Color(1.0f), glm::vec3(t));
}

int main()
{
	assert(SDL_Init(SDL_INIT_VIDEO) == 0);
	const int windowWidth = 1280;
	const int windowHeight = 720;
	const float viewportWidth = 10.0f * ((float)windowWidth / (float)windowHeight);
	const float viewportHeight = 10.0f;
	const float vocalLength = 10.0f;
	auto* window = SDL_CreateWindow("Raytracing", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, windowWidth, windowHeight, 0);
	assert(window != nullptr);

	auto *surface = SDL_GetWindowSurface(window);
	uint32_t* pixels = (uint32_t*)surface->pixels;
	
	// Draw

	SDL_LockSurface(surface);

	glm::vec3 camera(0.0f, 0.0f, 0.0f);
	Surface s(pixels, windowWidth, windowHeight);

	for (int i = 0; i < windowHeight; i++)
	{
		for (int j = 0; j < windowWidth; j++)
		{
			float u = (float)j / (float)windowWidth;
			float v = (float)i / (float)windowHeight;
			/*Color color;
			color.b = 0.25f;*/

			Ray ray{
				camera,
				(camera + glm::vec3(-viewportWidth / 2.0f, -viewportHeight / 2.0f, -vocalLength)) +
				glm::vec3(viewportWidth, 0.0f, 0.0f) * u +
				glm::vec3(0.0f, viewportHeight, 0.0f) * v -
				camera };
			s.setPixel(j, i, rayColor(ray));
		}
	}

	SDL_UnlockSurface(surface);
	SDL_UpdateWindowSurface(window);

	// End draw
	
	SDL_Event event;

	bool running = true;
	while (running)
	{
		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_QUIT)
			{
				running = false;
			}
		}
	}

	SDL_DestroyWindow(window);

	SDL_Quit();

	return 0;
}
