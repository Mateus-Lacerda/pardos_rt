#ifndef BOX_H
#define BOX_H

#include "hittable.h"
#include "material.h" // For GPUMaterial
#include <algorithm>

// GPU-compatible struct for a box (AABB).
struct GPUBox
{
    point3 box_min;
    point3 box_max;
    GPUMaterial material_data;
};

class box : public hittable
{
public:
    point3 box_min, box_max;
    shared_ptr<material> mat;

    box(const point3 &p0, const point3 &p1, shared_ptr<material> mat)
        : box_min(fmin_vec3(p0, p1)), box_max(fmax_vec3(p0, p1)), mat(mat) {}
};

#endif // BOX_H
