#include "HittableObjectsDevice.cuh"

#include "HittableObjects.cuh"
#include "Object/SphereObject.cuh"

HittableObjectsDevice::HittableObjectsDevice(int totalObject):
    m_totalObject(totalObject)
{
    cudaMalloc((void**)&m_objects, sizeof(IHittableObject*) * totalObject * 2);
    printf("create %p length %llu\n", m_objects, sizeof(IHittableObject*) * totalObject);
    printf("idx %p\n", &m_objects[1]);
}

void HittableObjectsDevice::hit(HitData& hitDataClosest, bool& hasValue, const Ray& ray, float tMin, float tMax) const
{
    float closestT = tMax;

    for (int i = 0; i < 2; i++)
    {

    }

    SphereObject oh_2(glm::vec3(-4.0f, 0.0f, -10.0f), 4.0f);

    bool objectHasValue = false;
    oh_2.hit(hitDataClosest, objectHasValue, ray, tMin, tMax);
    if (objectHasValue)
    {
        hasValue = true;
        closestT = hitDataClosest.t;
    }
}
