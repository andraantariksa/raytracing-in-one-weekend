#include "HittableObjectsDevice.cuh"

#include "Object/SphereObject.cuh"
#include "Object/BoxObject.cuh"

HittableObjectsDevice::HittableObjectsDevice(int totalObject):
    m_totalObject(totalObject)
{
    cudaMalloc(&m_objects, sizeof(IHittableObject*) * m_totalObject);
}

bool HittableObjectsDevice::hit(HitData& hitDataClosest, const Ray& ray, float tMin, float tMax) const
{
    float closestT = tMax;

    bool hasValue = false;

    for (int i = 0; i < m_totalObject; ++i)
    {
        if (m_objects[i]->hit(hitDataClosest, ray, tMin, closestT))
        {
            hasValue = true;
            closestT = hitDataClosest.t;
        }
    }
    return hasValue;
}
