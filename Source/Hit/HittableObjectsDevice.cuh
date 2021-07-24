#ifndef RAYTRACING_SOURCE_HIT_HITTABLEOBJECTSDEVICE_CUH
#define RAYTRACING_SOURCE_HIT_HITTABLEOBJECTSDEVICE_CUH

#include <optional>
#include <thrust/device_vector.h>

#include "IHittableObject.cuh"
#include "HittableObjects.cuh"

class HittableObjectsDevice
{
public:
    explicit HittableObjectsDevice(int totalObject);

    [[nodiscard]]
    __host__ __device__ void hit(HitData& hitData, bool& hasValue, const Ray& ray, float tMin, float tMax) const;

    friend class HittableObjects;
    IHittableObject** m_objects;
private:
    int m_totalObject;
};

#endif //RAYTRACING_SOURCE_HIT_HITTABLEOBJECTSDEVICE_CUH
