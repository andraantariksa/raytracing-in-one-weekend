#include <optional>

#include "SphereObject.cuh"

std::optional<HitData> SphereObject::hit(const Ray& ray, float t_min, float t_max) const
{
    // P = Point lies in the sphere surface (Vector)
    // C = m_center of sphere (Vector)
    // (x - C_x)^2 + (y - C_y)^2 + (z + C_z)^2 = r^2
    // To vector notation
    // (P - C) ⋅ (P - C) = r^2
    //
    // b = Direction of ray (Vector)
    // A = Ray origin or camera (Vector)
    // t = Scale of direction to touch the object that are hitable by  (Scalar)ray
    // P(t) = A + t * b
    //
    // (P(t) - C) ⋅ (P(t) - C) = r^2
    // (A + t * b - C) ⋅ (A + t * b - C) = r^2
    // (t * b + (A - C)) ⋅ (t * b + (A - C)) = r^2
    // (tb)^2 + t * b ⋅ (A - C) * 2 + (A - C) ⋅ (A - C) = r^2
    // t^2 * b ⋅ b + t * b ⋅ (A - C) * 2 + (A - C) ⋅ (A - C) - r^2 = 0
    // Every variable is known except t. Turn it into quadratic
    // (b ⋅ b) * t^2 + 2 * b ⋅ (A - C) * t + (A - C) ⋅ (A - C) - r^2 = 0
    //
    // ax^2 + bx + c
    // a = b ⋅ b
    // b = 2 * b ⋅ (A - C)
    // c = (A - C) ⋅ (A - C) - r^2
    auto aSubC = ray.origin() - m_center;
    auto a = glm::dot(ray.direction(), ray.direction());
//    auto b = 2.0f * glm::dot(ray.direction(), aSubC);
    auto halfB = glm::dot(ray.direction(), aSubC);
    auto c = glm::dot(aSubC, aSubC) - m_r * m_r;
//    auto discriminant = b * b - 4 * a * c;
    auto discriminant = halfB * halfB - a * c;
    if (discriminant < 0.0f)
    {
        return {};
    }

    float discriminantSqrt = std::sqrt(discriminant);

//        return (-b - std::sqrt(discriminant)) / (2.0f * a);
    auto root = (-halfB - discriminantSqrt) / a;
    if (root < t_min || root > t_max)
    {
        root = (-halfB + discriminantSqrt) / a;
        if (root < t_min || root > t_max)
        {
            return {};
        }
    }

    HitData hitData{};
    hitData.coord = ray.at(root),
    hitData.t = root;
    hitData.N = glm::normalize(hitData.coord - m_center);
    return hitData;
}
