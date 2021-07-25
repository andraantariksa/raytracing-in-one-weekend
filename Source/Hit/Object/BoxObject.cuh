#ifndef RAYTRACING_SOURCE_HIT_OBJECT_BOXOBJECT_CUH
#define RAYTRACING_SOURCE_HIT_OBJECT_BOXOBJECT_CUH

#include "../HitData.cuh"
#include "../../Ray.cuh"
#include "../IHittableObject.cuh"

class BoxObject: public IHittableObject
{
public:
    __host__ __device__ BoxObject(glm::vec3 center, glm::vec3 size, IMaterial* material):
        m_size(size),
        m_center(center),
        m_material(material)
    {
    }

    [[nodiscard]]
    __host__ __device__ bool hit(HitData& hitDataClosest, const Ray& ray, float t_min, float t_max) const override;
private:
    glm::vec3 m_size;
    glm::vec3 m_center;
    IMaterial* m_material;
};

#endif //RAYTRACING_SOURCE_HIT_OBJECT_BOXOBJECT_CUH
