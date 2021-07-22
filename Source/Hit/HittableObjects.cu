#include "HittableObjects.cuh"

void HittableObjects::hit(HitData& hitDataClosest, bool& hasValue, const Ray& ray, float tMin, float tMax) const
{
    float closestT = tMax;
    bool hitSomething = false;

    for (const auto& object : m_objects)
    {
//        auto hit = object->hit(ray, tMin, closestT);
//        if (hit.has_value())
        {
//            hitSomething = true;
//            hitData = hit.value();
//            closestT = hitData.value().t;
        }
    }

    if (!hitSomething)
    {
//        return {};
    }

//    return hitData;
}
