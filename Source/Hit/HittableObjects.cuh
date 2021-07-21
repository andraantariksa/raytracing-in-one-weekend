#ifndef RAYTRACING_SOURCE_HIT_HITTABLEOBJECTS_CUH
#define RAYTRACING_SOURCE_HIT_HITTABLEOBJECTS_CUH

#include <memory>
#include <vector>

#include "IHittableObject.cu"

class HittableObjects: public IHittableObject
{
public:
    [[nodiscard]]
    __host__ __device__ std::optional<HitData> hit(const Ray& ray, float tMin, float tMax) const override;

    void add(const std::shared_ptr<IHittableObject>& hittableObject) { m_objects.push_back(hittableObject); }
    void clear() { m_objects.clear(); }
private:
    std::vector<std::shared_ptr<IHittableObject>> m_objects;
};

#endif //RAYTRACING_SOURCE_HIT_HITTABLEOBJECTS_CUH
