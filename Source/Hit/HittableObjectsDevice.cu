#include "HittableObjectsDevice.cuh"

#include "HittableObjects.cuh"
#include "Object/SphereObject.cuh"
#include "Object/BoxObject.cuh"

HittableObjectsDevice::HittableObjectsDevice(int totalObject):
    m_totalObject(totalObject)
{
    cudaMalloc((void**)&m_objects, sizeof(IHittableObject*) * totalObject * 2);
}

void HittableObjectsDevice::hit(HitData& hitDataClosest, bool& hasValue, const Ray& ray, float tMin, float tMax) const
{
    float closestT = tMax;

    SphereObject oh_2(glm::vec3(0.0f, 0.0f, -10.0f), 4.0f);
    SphereObject oh_1(glm::vec3(0.0f, -104.0f, -10.0f), 100.0f);
    BoxObject oh_3(glm::vec3(3.0f, 3.0f, -10.0f), glm::vec3(1.0f));

    bool objectHasValue = false;
    oh_1.hit(hitDataClosest, objectHasValue, ray, tMin, tMax);
    if (objectHasValue)
    {
        hasValue = true;
        closestT = hitDataClosest.t;
    }

    objectHasValue = false;
    oh_2.hit(hitDataClosest, objectHasValue, ray, tMin, tMax);
    if (objectHasValue)
    {
        hasValue = true;
        closestT = hitDataClosest.t;
    }
}
