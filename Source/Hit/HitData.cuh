#ifndef RAYTRACING_SOURCE_HITDATA_CUH
#define RAYTRACING_SOURCE_HITDATA_CUH

#include <glm/glm.hpp>
#include <thrust/device_ptr.h>

#include "../Material/IMaterial.cuh"

class IMaterial;

class HitData
{
public:
    glm::vec3 N;
    float t;
    glm::vec3 coord;
    IMaterial* material;

    __host__ __device__ HitData() = default;

    __host__ __device__ HitData(const HitData& other):
        N(other.N),
        t(other.t),
        coord(other.coord)
    {
    }
};

#endif //RAYTRACING_SOURCE_HITDATA_HPP
