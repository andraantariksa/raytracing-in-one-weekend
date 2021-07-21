#ifndef RAYTRACING_SOURCE_MISC_RANDOM_CUH
#define RAYTRACING_SOURCE_MISC_RANDOM_CUH

#include <random>

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

#endif //RAYTRACING_SOURCE_MISC_RANDOM_CUH
