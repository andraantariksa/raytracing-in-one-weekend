//
// Created by andra on 24/07/2021.
//

#ifndef RAYTRACING_SOURCE_MATERIAL_METAL_CUH
#define RAYTRACING_SOURCE_MATERIAL_METAL_CUH

#include "../Hit/HitData.cuh"
#include "../Ray.cuh"

class Metal: public IMaterial
{
public:
    bool scatter(HitData& hitData, const Ray& ray, Color& attenuation, Ray& scatteredRay) const override;
};

#endif //RAYTRACING_SOURCE_MATERIAL_METAL_CUH
