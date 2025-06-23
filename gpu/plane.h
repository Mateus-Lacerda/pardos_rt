#ifndef PLANE_H
#define PLANE_H

#include "hittable.h"
#include "material.h" // For GPUMaterial
#include "vec3.h"     // For point3, vec3

// GPU-compatible struct for a plane.
// This is a plain data struct that can be copied to the GPU.
struct GPUPlane
{
    point3 point;
    vec3 normal;
    GPUMaterial material_data;
};

class plane : public hittable
{
public:
    point3 point;
    vec3 normal;
    shared_ptr<material> mat;

    plane(const point3 &point, const vec3 &normal, shared_ptr<material> mat)
        : point(point), normal(unit_vector(normal)), mat(mat) {}
};

#endif // PLANE_H
