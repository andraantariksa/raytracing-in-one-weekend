#ifndef RAYTRACING_SOURCE_HIT_OBJECT_BOXOBJECT_CUH
#define RAYTRACING_SOURCE_HIT_OBJECT_BOXOBJECT_CUH

#include "../HitData.cuh"
#include "../../Ray.cuh"
#include "../IHittableObject.cuh"

class BoxObject: public IHittableObject
{
public:
    __host__ __device__ BoxObject(glm::vec3 center, glm::vec3 size): m_size(size), m_center(center)
    {
    }

    [[nodiscard]]
    __host__ __device__ void hit(HitData& hitDataClosest, bool& hasValue, const Ray& ray, float t_min, float t_max) const override;
private:
    glm::vec3 m_size;
    glm::vec3 m_center;
};

#endif //RAYTRACING_SOURCE_HIT_OBJECT_BOXOBJECT_CUH
