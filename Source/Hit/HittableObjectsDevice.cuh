#ifndef RAYTRACING_SOURCE_HIT_HITTABLEOBJECTSDEVICE_CUH
#define RAYTRACING_SOURCE_HIT_HITTABLEOBJECTSDEVICE_CUH

#include "IHittableObject.cuh"

class HittableObjectsDevice
{
public:
    explicit HittableObjectsDevice(int totalObject);

    [[nodiscard]]
    __host__ __device__ bool hit(HitData& hitData, const Ray& ray, float tMin, float tMax) const;

    IHittableObject** getObjects() const { return m_objects; }
    int getTotalObject() const { return m_totalObject; }
private:
    IHittableObject** m_objects;
    int m_totalObject;
};

#endif //RAYTRACING_SOURCE_HIT_HITTABLEOBJECTSDEVICE_CUH
