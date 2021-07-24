#ifndef RAYTRACING_SOURCE_IMATERIAL_CUH
#define RAYTRACING_SOURCE_IMATERIAL_CUH

#include "../Hit/HitData.cuh"
#include "../Ray.cuh"
#include "../Typedef.cuh"

class HitData;

class IMaterial
{
public:
    virtual bool scatter(HitData& hitData, const Ray& ray, Color& attenuation, Ray& scatteredRay) const = 0;
};

#endif //RAYTRACING_SOURCE_IMATERIAL_CUH
