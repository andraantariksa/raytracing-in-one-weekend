#ifndef RAYTRACING_SOURCE_MATERIAL_LAMBERTIAN_CUH
#define RAYTRACING_SOURCE_MATERIAL_LAMBERTIAN_CUH

#include "../Typedef.cuh"
#include "IMaterial.cuh"

class Lambertian : public IMaterial
{
public:
    __device__ Lambertian(const Color& color) :
        m_albedo(color)
    {
    }

    __device__ bool scatter(HitData& hitData,
        const Ray& ray,
        Color& attenuation,
        Ray& scatteredRay,
        curandState* localRandomState,
        int framebufferIdx) const override;
private:
    Color m_albedo;
};

#endif //RAYTRACING_SOURCE_MATERIAL_LAMBERTIAN_CUH
