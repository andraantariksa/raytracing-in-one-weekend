#ifndef RAYTRACING_SOURCE_IHITTABLEOBJECT_HPP
#define RAYTRACING_SOURCE_IHITTABLEOBJECT_HPP

#include <optional>

#include "../Ray.hpp"
#include "HitData.hpp"

class IHittableObject
{
public:
    [[nodiscard]]
    virtual std::optional<HitData> hit(const Ray& ray, float tMin, float tMax) const = 0;
};

#endif //RAYTRACING_SOURCE_IHITTABLEOBJECT_HPP
