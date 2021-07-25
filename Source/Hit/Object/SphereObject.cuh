#ifndef RAYTRACING_SOURCE_HIT_OBJECT_SPHEREOBJECT_CUH
#define RAYTRACING_SOURCE_HIT_OBJECT_SPHEREOBJECT_CUH

#include "../HitData.cuh"
#include "../IHittableObject.cuh"

class SphereObject: public IHittableObject
{
public:
    __host__ __device__ SphereObject(glm::vec3 center, float r, IMaterial* material):
        m_r(r),
        m_center(center),
        m_material(material)
    {
    }

    [[nodiscard]]
    __host__ __device__ bool hit(HitData& hitDataClosest, const Ray& ray, float t_min, float t_max) const override;
private:
    float m_r;
    glm::vec3 m_center;
    IMaterial* m_material;
};

#endif //RAYTRACING_SOURCE_OBJECT_SPHEREOBJECT_HPP
