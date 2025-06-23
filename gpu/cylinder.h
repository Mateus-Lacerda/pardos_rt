#ifndef CYLINDER_H
#define CYLINDER_H

#include "hittable.h"
#include "material.h" // For GPUMaterial
#include <cmath>

// GPU-compatible struct for a cylinder (aligned with Y-axis).
struct GPUCylinder
{
    point3 center;
    double radius;
    double half_height;
    GPUMaterial material_data;
};

class cylinder : public hittable
{
public:
    point3 center;
    double radius, half_height;
    shared_ptr<material> mat;

    // Constructor takes full height and stores half_height for symmetrical calculations
    cylinder(const point3 &center, double radius, double height, shared_ptr<material> mat)
        : center(center), radius(radius), half_height(height / 2.0), mat(mat) {}
};

#endif // CYLINDER_H
