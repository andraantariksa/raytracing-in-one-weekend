#include "Camera.hpp"

Camera::Camera(glm::vec3 origin, float viewportWidth, float viewportHeight, float vocalLength):
    m_origin(origin),
    m_viewportWidth(viewportWidth),
    m_viewportHeight(viewportHeight),
    m_vocalLength(vocalLength)
{
    recalculate();
}

void Camera::recalculate()
{
    m_viewportOrigin = m_origin + glm::vec3(-m_viewportWidth / 2.0f, -m_viewportHeight / 2.0f, -m_vocalLength);
}

Ray Camera::getRay(float u, float v)
{
    auto rayDirection = m_viewportOrigin +
        glm::vec3(m_viewportWidth, 0.0f, 0.0f) * u +
        glm::vec3(0.0f, m_viewportHeight, 0.0f) * v -
        m_origin;
    return Ray(m_origin,rayDirection);
}

void Camera::transform(glm::mat4 transformMat)
{
    m_origin = transformMat * glm::vec4(m_origin, 1.0f);
    recalculate();
}
