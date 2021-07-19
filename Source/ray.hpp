#include <glm/glm.hpp>

class Ray
{
public:
	Ray(glm::vec3& origin, glm::vec3& direction):
		m_origin(origin),
		m_direction(direction)
	{
	};

	glm::vec3 coordAt(float t) { return m_origin + m_direction * t; }

	glm::vec3 origin() const { return m_origin; };
	glm::vec3 direction() const { return m_direction; };
private:
	glm::vec3 m_origin;
	glm::vec3 m_direction;
};
