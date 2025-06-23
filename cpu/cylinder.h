#ifndef CYLINDER_H
#define CYLINDER_H

#include "hittable.h"
#include "material.h"
#include <cmath>

class cylinder : public hittable
{
public:
    point3 base_center;
    double radius, height;
    shared_ptr<material> mat;

    cylinder(const point3 &base_center, double radius, double height, shared_ptr<material> mat)
        : base_center(base_center), radius(radius), height(height), mat(mat) {}

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override
    {
        // Cylinder aligned with y axis
        auto oc = r.origin() - base_center;
        auto a = r.direction().x() * r.direction().x() + r.direction().z() * r.direction().z();
        auto b = 2.0 * (oc.x() * r.direction().x() + oc.z() * r.direction().z());
        auto c = oc.x() * oc.x() + oc.z() * oc.z() - radius * radius;
        auto discriminant = b * b - 4 * a * c;
        if (discriminant < 0)
            return false;
        auto sqrtd = std::sqrt(discriminant);
        auto t0 = (-b - sqrtd) / (2 * a);
        auto t1 = (-b + sqrtd) / (2 * a);
        for (auto t : {t0, t1})
        {
            if (ray_t.contains(t))
            {
                auto y = oc.y() + t * r.direction().y();
                if (y >= 0 && y <= height)
                {
                    rec.t = t;
                    rec.p = r.at(t);
                    vec3 outward_normal = vec3(rec.p.x() - base_center.x(), 0, rec.p.z() - base_center.z()) / radius;
                    rec.set_face_normal(r, outward_normal);
                    rec.mat = mat;
                    return true;
                }
            }
        }
        return false;
    }
};

#endif // CYLINDER_H
