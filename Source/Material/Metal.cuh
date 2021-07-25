#ifndef RAYTRACING_SOURCE_MATERIAL_METAL_CUH
#define RAYTRACING_SOURCE_MATERIAL_METAL_CUH

#include "../Hit/HitData.cuh"
#include "../Ray.cuh"

class Metal : public IMaterial
{
public:
    __device__ Metal(const Color& albedo):
        m_albedo(albedo)
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

#endif //RAYTRACING_SOURCE_MATERIAL_METAL_CUH
