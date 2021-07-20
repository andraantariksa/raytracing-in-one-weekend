#ifndef RAYTRACING_SOURCE_RAY_HPP
#define RAYTRACING_SOURCE_RAY_HPP

#include <glm/glm.hpp>

class Ray
{
public:
	Ray(glm::vec3& origin, glm::vec3& direction):
		m_origin(origin),
		m_direction(direction)
	{
	}

	[[nodiscard]]
	glm::vec3 at(float t) const { return m_origin + m_direction * t; }

    [[nodiscard]]
	glm::vec3 origin() const { return m_origin; };
    [[nodiscard]]
	glm::vec3 direction() const { return m_direction; };
private:
	glm::vec3 m_origin;
	glm::vec3 m_direction;
};

#endif
