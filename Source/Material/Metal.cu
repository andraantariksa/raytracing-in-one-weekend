#include "Metal.cuh"

__device__ bool Metal::scatter(
    HitData& hitData,
    const Ray& ray,
    Color& attenuation,
    Ray& scatteredRay,
    curandState* localRandomState,
    int framebufferIdx) const
{
    auto reflected = glm::reflect(glm::normalize(ray.direction()), hitData.N);
    scatteredRay = Ray(hitData.coord, reflected);
    attenuation = m_albedo;
    return glm::dot(scatteredRay.direction(), hitData.N) > 0.0f;
}
