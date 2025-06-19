#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "rtweekend.h"
#include "vec3.h"

class sphere : public hittable {
public:
    sphere(const point3& center, double radius, shared_ptr<material> mat) 
    : center(center), radius(std::fmax(0,radius)), mat(mat) {}

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        vec3 oc = center - r.origin();
        auto a = r.direction().lenght_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.lenght_squared() - radius*radius;

        auto discriminant = h*h - a*c;
        if (discriminant < 0)
            return false;

        auto sqrtd = std::sqrt(discriminant);

        // Nearest root that lies in the acceptable range
        auto root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            // If the root is not in range, try the next
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                // None of roots are in range
                return false;
        }

        // Store the hit record
        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        // Sets the normal of the record and determines if its a front facing ray or not
        rec.set_face_normal(r, outward_normal);
        // rec.normal = (rec.p - center) / radius;
        rec.mat = mat;

        return true;
    }

private:
    point3 center;
    double radius;
    shared_ptr<material> mat;
};

#endif // !SPHERE_H
