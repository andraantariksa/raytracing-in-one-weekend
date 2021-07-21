#include "HittableObjects.cuh"

std::optional<HitData> HittableObjects::hit(const Ray& ray, float tMin, float tMax) const
{
    float closestT = tMax;
    HitData hitData{};
    bool hitSomething = false;

    for (const auto& object : m_objects)
    {
        auto hit = object->hit(ray, tMin, closestT);
        if (hit.has_value())
        {
            hitSomething = true;
            hitData = hit.value();
            closestT = hitData.t;
        }
    }

    if (!hitSomething)
    {
        return {};
    }

    return hitData;
}
