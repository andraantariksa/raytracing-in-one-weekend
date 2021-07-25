#ifndef RAYTRACING_SOURCE_MISC_RANDOM_CUH
#define RAYTRACING_SOURCE_MISC_RANDOM_CUH

#include "../../../../../../Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30037/include/random"
#include "../../External/glm/glm/gtx/norm.hpp"
#include "../../External/glm/glm/glm.hpp"

inline float randomFloat()
{
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    static std::mt19937 generator;
    return distribution(generator);
}

inline int randomIntOne()
{
    static std::uniform_int_distribution<int> distribution(-1, 1);
    static std::mt19937 generator;
    return distribution(generator);
}

__host__ __device__ inline float goldenNoise(const glm::vec2& coord, float seed)
{
    float _ignore;
    return std::modf(std::tan(glm::distance(coord * 1.61803398874989484820459f, coord) * seed) * coord.x, &_ignore);
}

__host__ __device__ inline float interleavedGradientNoise(const glm::vec2& coord)
{
    float _ignore;
    return std::modf(52.9829189f * std::modf(coord.x * 0.06711056f + coord.y * 0.00583715f, &_ignore), &_ignore);
}

__device__ inline glm::vec3 randomVec3(curandState* localRandomState, int framebufferIdx)
{
    return 2.0f * glm::vec3(curand_uniform(localRandomState), curand_uniform(localRandomState), curand_uniform(localRandomState)) - 1.0f;
}

__device__ inline glm::vec3 randomUnitVector(curandState* localRandomState, int framebufferIdx)
{
    return glm::normalize(randomVec3(localRandomState, framebufferIdx));
}

__device__ inline glm::vec3 randomInUnitSphere(curandState* localRandomState, int framebufferIdx)
{
    while (true)
    {
        auto v = randomVec3(localRandomState, framebufferIdx);
        if (glm::length2(v) >= 1.0f)
        {
            continue;
        }
        return v;
    }
}

__device__ inline glm::vec3 randomInHemisphere(glm::vec3& normal, curandState* localRandomState, int framebufferIdx)
{
    glm::vec3 inUnitSphere = randomInUnitSphere(localRandomState, framebufferIdx);
    if (glm::dot(inUnitSphere, normal) > 0.0f)
    {
        return inUnitSphere;
    }
    else
    {
        return -inUnitSphere;
    }
}


#endif //RAYTRACING_SOURCE_MISC_RANDOM_CUH
