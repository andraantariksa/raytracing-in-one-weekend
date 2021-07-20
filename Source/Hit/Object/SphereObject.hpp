#ifndef RAYTRACING_SOURCE_OBJECT_SPHEREOBJECT_HPP
#define RAYTRACING_SOURCE_OBJECT_SPHEREOBJECT_HPP

#include <optional>

#include "../HitData.hpp"
#include "../IHittableObject.hpp"

class SphereObject: public IHittableObject
{
public:
    SphereObject(glm::vec3 center, float r): m_r(r), m_center(center)
    {
    }

    [[nodiscard]]
    std::optional<HitData> hit(const Ray &ray, float t_min, float t_max) const override;
private:
    float m_r;
    glm::vec3 m_center;
};

#endif //RAYTRACING_SOURCE_OBJECT_SPHEREOBJECT_HPP
