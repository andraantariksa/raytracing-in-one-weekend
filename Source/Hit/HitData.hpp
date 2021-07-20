#ifndef RAYTRACING_SOURCE_HITDATA_HPP
#define RAYTRACING_SOURCE_HITDATA_HPP

#include <glm/glm.hpp>

struct HitData
{
    glm::vec3 N;
    float t;
    glm::vec3 coord;
};

#endif //RAYTRACING_SOURCE_HITDATA_HPP
