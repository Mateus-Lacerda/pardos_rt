#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray
{
public:
    __host__ __device__ ray() {}
    // <<< FIX: Adicionado o construtor com o tempo e o membro tm >>>
    __host__ __device__ ray(const point3 &origin, const vec3 &direction, double time = 0.0)
        : orig(origin), dir(direction), tm(time) {}

    __host__ __device__ point3 origin() const { return orig; }
    __host__ __device__ vec3 direction() const { return dir; }
    __host__ __device__ double time() const { return tm; } // <<< FIX: Adicionado método time()

    __host__ __device__ point3 at(double t) const
    {
        return orig + t * dir;
    }

private:
    point3 orig;
    vec3 dir;
    double tm; // <<< FIX: Adicionado membro de tempo
};

#endif