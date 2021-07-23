#ifndef RAYTRACING_SOURCE_MISC_RANDOM_CUH
#define RAYTRACING_SOURCE_MISC_RANDOM_CUH

#include <random>
#include <glm/glm.hpp>

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

__host__ __device__ inline float interleavedGradientNoise(const glm::vec2& coord) {
    float _ignore;
    return std::modf(52.9829189f * std::modf(coord.x * 0.06711056f + coord.y * 0.00583715f, &_ignore), &_ignore);
}

#endif //RAYTRACING_SOURCE_MISC_RANDOM_CUH
