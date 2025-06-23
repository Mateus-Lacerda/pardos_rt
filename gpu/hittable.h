#ifndef HITTABLE_H
#define HITTABLE_H

#include "rtweekend.h"
#include <string>
#include <memory> // For shared_ptr

// Forward declare the host-side material class (material.h will define it)
class material;

// hit_record struct for HOST-SIDE use
// It uses shared_ptr<material> for polymorphism on the CPU.
struct hit_record
{
    point3 p;
    vec3 normal;
    std::shared_ptr<material> mat; // <--- This is now shared_ptr for host
    double t;
    bool front_face;

    // Explicit default constructor, as shared_ptr might make it non-trivial
    __host__ hit_record() : p(), normal(), mat(nullptr), t(0.0), front_face(false) {}

    // This method is for host-side. No __device__ here.
    void set_face_normal(const ray &r, const vec3 &outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// Forward declarations for GPU-compatible structs (defined in material.h and other primitive headers)
// These are not needed directly in hittable.h anymore if hit_record is host-only.
// But the classes like sphere.h still use them in their GPU structs.
// So these forward declarations are only needed by CUDA compilation units.

// Host-side polymorphic hittable base class
class hittable
{
public:
    hittable() : _id(std::to_string(_next_id++)) {}

    virtual ~hittable() = default;

    virtual bool hit(const ray &r, interval ray_t, hit_record &rec) const = 0; // Uses host-side hit_record

    std::string id() const { return _id; }

private:
    std::string _id;
    static inline size_t _next_id = 0;
};

// **IMPORTANT: No #include "material.h" here!**
// This avoids the circular dependency.

#endif // !HITTABLE_H