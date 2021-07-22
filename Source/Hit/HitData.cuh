#ifndef RAYTRACING_SOURCE_HITDATA_HPP
#define RAYTRACING_SOURCE_HITDATA_HPP

#include <glm/glm.hpp>

class HitData
{
public:
    glm::vec3 N;
    float t;
    glm::vec3 coord;

    __host__ __device__ HitData() = default;

    __host__ __device__ HitData(const HitData& other):
        N(other.N),
        t(other.t),
        coord(other.coord)
    {
    }
};

#endif //RAYTRACING_SOURCE_HITDATA_HPP
