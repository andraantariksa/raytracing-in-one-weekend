#ifndef RAYTRACING_SOURCE_IHITTABLEOBJECT_HPP
#define RAYTRACING_SOURCE_IHITTABLEOBJECT_HPP

#include <optional>

#include "../Ray.cuh"
#include "HitData.cuh"

class IHittableObject
{
public:
    [[nodiscard]]
    __host__ __device__ virtual void hit(HitData& hitDataClosest, bool& hasValue, const Ray& ray, float tMin, float tMax) const = 0;
};

#endif //RAYTRACING_SOURCE_IHITTABLEOBJECT_HPP
