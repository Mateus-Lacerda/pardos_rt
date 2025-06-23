#ifndef PLANE_H
#define PLANE_H

#include "hittable.h"
#include "material.h"

class plane : public hittable
{
public:
    point3 p0;
    vec3 normal;
    shared_ptr<material> mat;

    plane(const point3 &p0, const vec3 &normal, shared_ptr<material> mat)
        : p0(p0), normal(unit_vector(normal)), mat(mat) {}

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override
    {
        auto denom = dot(normal, r.direction());
        if (fabs(denom) > 1e-6)
        {
            auto t = dot(p0 - r.origin(), normal) / denom;
            if (ray_t.contains(t))
            {
                rec.t = t;
                rec.p = r.at(t);
                rec.set_face_normal(r, normal);
                rec.mat = mat;
                return true;
            }
        }
        return false;
    }
};

#endif // PLANE_H
