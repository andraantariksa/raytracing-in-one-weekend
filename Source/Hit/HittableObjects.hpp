#ifndef RAYTRACING_SOURCE_HIT_HITTABLEOBJECTS_HPP
#define RAYTRACING_SOURCE_HIT_HITTABLEOBJECTS_HPP

#include <memory>
#include <vector>

#include "IHittableObject.hpp"

class HittableObjects: public IHittableObject
{
public:
    [[nodiscard]]
    std::optional<HitData> hit(const Ray& ray, float tMin, float tMax) const override;

    void add(const std::shared_ptr<IHittableObject>& hittableObject) { m_objects.push_back(hittableObject); }
    void clear() { m_objects.clear(); }
private:
    std::vector<std::shared_ptr<IHittableObject>> m_objects;
};

#endif //RAYTRACING_SOURCE_HIT_HITTABLEOBJECTS_HPP
