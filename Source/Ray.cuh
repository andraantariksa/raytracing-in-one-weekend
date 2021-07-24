#ifndef RAYTRACING_SOURCE_RAY_CUH
#define RAYTRACING_SOURCE_RAY_CUH

#include <glm/glm.hpp>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

class Ray
{
public:
    __host__ __device__ Ray(const glm::vec3& origin, const glm::vec3& direction):
		m_origin(origin),
		m_direction(direction)
	{
	}

	[[nodiscard]]
    __host__ __device__ glm::vec3 at(float t) const { return m_origin + m_direction * t; }

    [[nodiscard]]
    __host__ __device__ glm::vec3 origin() const { return m_origin; };
    [[nodiscard]]
    __host__ __device__ glm::vec3 direction() const { return m_direction; };
private:
	glm::vec3 m_origin;
	glm::vec3 m_direction;
};

#endif
