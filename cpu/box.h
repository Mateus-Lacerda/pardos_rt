#ifndef BOX_H
#define BOX_H

#include "hittable.h"
#include "material.h"
#include <algorithm>

class box : public hittable
{
public:
    point3 min_corner, max_corner;
    shared_ptr<material> mat;

    box(const point3 &min_corner, const point3 &max_corner, shared_ptr<material> mat)
        : min_corner(min_corner), max_corner(max_corner), mat(mat) {}

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override
    {
        double tmin = ray_t.min, tmax = ray_t.max;
        for (int a = 0; a < 3; a++)
        {
            double invD = 1.0 / r.direction()[a];
            double t0 = (min_corner[a] - r.origin()[a]) * invD;
            double t1 = (max_corner[a] - r.origin()[a]) * invD;
            if (invD < 0.0)
                std::swap(t0, t1);
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin)
                return false;
        }
        rec.t = tmin;
        rec.p = r.at(rec.t);
        // Determina a normal baseada na face atingida
        vec3 outward_normal(0, 0, 0);
        for (int a = 0; a < 3; a++)
        {
            if (fabs(rec.p[a] - min_corner[a]) < 1e-4)
                outward_normal[a] = -1;
            else if (fabs(rec.p[a] - max_corner[a]) < 1e-4)
                outward_normal[a] = 1;
        }
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;
        return true;
    }
};

#endif // BOX_H
