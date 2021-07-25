#ifndef RAYTRACING_SOURCE_IHITTABLEOBJECT_CUH
#define RAYTRACING_SOURCE_IHITTABLEOBJECT_CUH

#include <optional>

#include "../Ray.cuh"
#include "HitData.cuh"

class IHittableObject
{
public:
    [[nodiscard]]
    __host__ __device__ virtual bool hit(HitData& hitDataClosest, const Ray& ray, float tMin, float tMax) const = 0;
};

#endif //RAYTRACING_SOURCE_IHITTABLEOBJECT_HPP
