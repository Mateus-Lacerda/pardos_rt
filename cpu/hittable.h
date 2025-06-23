#ifndef HITTABLE_H
#define HITTABLE_H

#include "rtweekend.h"
#include <string>

class material;

class hit_record {
public:
    point3 p;
    vec3 normal;
    shared_ptr<material> mat;
    double t;
    bool front_face;

    void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit lenght.
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


class hittable {
public:
    hittable() : _id(std::to_string(_next_id++)) {}

    virtual ~hittable() = default;

    virtual bool hit(const ray& r, interval ray, hit_record& rec) const = 0;

    std::string id() const { return _id; }

private:
    std::string _id;
    static inline size_t _next_id = 0;
};

#endif // !HITTABLE_H
