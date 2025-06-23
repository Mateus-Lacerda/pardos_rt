#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "material.h" // For GPUMaterial
#include "rtweekend.h"
#include "vec3.h"

// GPU-compatible struct for a sphere.
// This is a plain data struct that can be copied to the GPU.
struct GPUSphere
{
    point3 center;
    double radius;
    GPUMaterial material_data;
    bool is_moving;
    vec3 center_vec;
};

class sphere : public hittable
{
public:
    point3 center;
    double radius;
    shared_ptr<material> mat;
    bool is_moving;
    vec3 center_vec;

    // Constructor for static spheres
    sphere(const point3 &center, double radius, shared_ptr<material> mat)
        : center(center), radius(std::fmax(0, radius)), mat(mat), is_moving(false) {}

    // Constructor for moving spheres
    sphere(const point3 &center1, const point3 &center2, double radius, shared_ptr<material> mat)
        : center(center1), radius(std::fmax(0, radius)), mat(mat), is_moving(true)
    {
        center_vec = center2 - center1;
    }

    void move(char &movement)
    {
        switch (movement)
        {
        case 'w':
            center = center + point3(0, 0, -0.01);
            break;
        case 'a':
            center = center + point3(-0.01, 0, 0);
            break;
        case 's':
            center = center + point3(0, 0, 0.01);
            break;
        case 'd':
            center = center + point3(0.01, 0, 0);
            break;
        case 'j':
            center = center + point3(0, 0.01, 0);
            break;
        case 'k':
            center = center + point3(0, -0.01, 0);
            break;
        }
    }
};

#endif // !SPHERE_H
