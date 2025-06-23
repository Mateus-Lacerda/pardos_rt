#ifndef HITTABLE_H
#define HITTABLE_H

#include "rtweekend.h"
#include "material.h" // For GPUMaterial struct

// hit_record struct for BOTH host and device use.
// It uses GPUMaterial, a flat data struct, making it compatible with CUDA kernels.
struct hit_record
{
    point3 p;
    vec3 normal;
    GPUMaterial mat; // <--- Changed from shared_ptr<material> to GPUMaterial
    double t;
    bool front_face;

    // Default constructor needs to be host/device compatible
    __host__ __device__ hit_record() : p(), normal(), mat(), t(0.0), front_face(false) {}

    // This method is now __host__ __device__
    __host__ __device__ void set_face_normal(const ray &r, const vec3 &outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// Host-side polymorphic hittable base class
class hittable
{
public:
    virtual ~hittable() = default;
};

#endif // !HITTABLE_H