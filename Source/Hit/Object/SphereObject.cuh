#ifndef RAYTRACING_SOURCE_OBJECT_SPHEREOBJECT_HPP
#define RAYTRACING_SOURCE_OBJECT_SPHEREOBJECT_HPP

#include "../HitData.cuh"
#include "../IHittableObject.cuh"

class SphereObject: public IHittableObject
{
public:
    __host__ __device__ SphereObject(glm::vec3 center, float r): m_r(r), m_center(center)
    {
    }

    [[nodiscard]]
    __host__ __device__ void hit(HitData& hitDataClosest, bool& hasValue, const Ray& ray, float t_min, float t_max) const override;
private:
    float m_r;
    glm::vec3 m_center;
};

#endif //RAYTRACING_SOURCE_OBJECT_SPHEREOBJECT_HPP
