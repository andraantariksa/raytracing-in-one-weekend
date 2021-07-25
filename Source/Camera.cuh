#ifndef RAYTRACING_SOURCE_CAMERA_CUH
#define RAYTRACING_SOURCE_CAMERA_CUH

#include "Ray.cuh"
#include <glm/glm.hpp>

class Camera
{
public:
    Camera(glm::vec3& origin, glm::vec3& lookAt, glm::vec3& up, float verticalFieldOfView, float aspectRatio);

    __host__ __device__ Ray getRay(float u, float v) const;
    void transform(glm::mat4 transformMat);
    void recalculate();
private:
    float m_viewportWidth;
    float m_viewportHeight;
    float m_verticalFieldOfView;
    float m_aspectRatio;
    glm::vec3 m_vertical;
    glm::vec3 m_horizontal;
    glm::vec3 m_origin;
    glm::vec3 m_viewportOrigin;
    glm::vec3 m_up;
    glm::vec3 m_lookAt;
};

#endif //RAYTRACING_SOURCE_CAMERA_CUH
