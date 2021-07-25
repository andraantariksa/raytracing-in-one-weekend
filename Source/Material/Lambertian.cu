#include "Lambertian.cuh"

#include "../Util/Random.cuh"
#include "../Ray.cuh"
#include <glm/glm.hpp>

__device__ bool Lambertian::scatter(
    HitData& hitData,
    const Ray& ray,
    Color& attenuation,
    Ray& scatteredRay,
    curandState* localRandomState,
    int framebufferIdx) const
{
    auto scatterDirection = hitData.N + randomUnitVector(localRandomState, framebufferIdx);
    if (glm::all(glm::lessThan(glm::abs(scatterDirection), glm::vec3(std::numeric_limits<float>::epsilon()))))
    {
        scatterDirection = hitData.N;
    }
    scatteredRay = Ray(hitData.coord, scatterDirection);
    attenuation = m_albedo;
    return true;
}