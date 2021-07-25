#include "BoxObject.cuh"

#include <glm/glm.hpp>

#include "../../Ray.cuh"

bool BoxObject::hit(HitData& hitDataClosest, const Ray& ray, float t_min, float t_max) const
{
    auto m = 1.0f / ray.direction();
    auto n = m * ray.origin();
    auto k = glm::abs(m) * m_size;
    auto t1 = -n - k;
    auto t2 = -n + k;
    auto tN = std::max(std::max(t1.x, t1.y), t1.z);
    auto tF = std::min(std::min(t2.x, t2.y), t2.z);
    if (tN > tF || tF < 0.0f)
    {
        return false;
    }
    hitDataClosest.coord = ray.at(tN);
    hitDataClosest.t = tN;
    glm::vec3 a(t1.yzx);
    glm::vec3 b(t1.xyz);
    glm::vec3 c(t1.zxy);
    hitDataClosest.N = -glm::sign(ray.direction()) * glm::step(a, b) * glm::step(c, b);
    return true;
}