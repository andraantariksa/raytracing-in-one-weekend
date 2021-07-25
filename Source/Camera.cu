#include "Ray.cuh"
#include "Camera.cuh"

Camera::Camera(glm::vec3& origin, glm::vec3& lookAt, glm::vec3& up, float verticalFieldOfView, float aspectRatio) :
    m_origin(origin),
    m_up(up),
    m_lookAt(lookAt),
    m_verticalFieldOfView(verticalFieldOfView),
    m_aspectRatio(aspectRatio)
{
    recalculate();
}

void Camera::recalculate()
{
    auto h = std::tan(m_verticalFieldOfView / 2.0f);

    m_viewportHeight = 2.0f * h;
    m_viewportWidth = m_aspectRatio * m_viewportHeight;

    auto w = glm::normalize(m_origin - m_lookAt);
    auto u = glm::normalize(glm::cross(m_up, w));
    auto v = glm::cross(w, u);

    m_horizontal = u * m_viewportWidth;
    m_vertical = v * m_viewportHeight;
    m_viewportOrigin = m_origin - (m_horizontal / 2.0f) - (m_vertical / 2.0f) - w;
}

Ray Camera::getRay(float s, float t) const
{
    auto rayDirection = m_viewportOrigin +
        m_horizontal * s +
        m_vertical * t -
        m_origin;
    return Ray(m_origin, rayDirection);
}

void Camera::transform(glm::mat4 transformMat)
{
    m_origin = transformMat * glm::vec4(m_origin, 1.0f);
    recalculate();
}
