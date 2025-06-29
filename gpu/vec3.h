#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>
#include <ostream>

// Forward declarations for host-only random_double from rtweekend.h
#ifndef __CUDACC__
inline double random_double();
inline double random_double(double min, double max);
#endif

class vec3
{
public:
    double e[3];

    __host__ __device__ vec3() : e{0, 0, 0} {}
    __host__ __device__ vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    __host__ __device__ double x() const { return e[0]; }
    __host__ __device__ double y() const { return e[1]; }
    __host__ __device__ double z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ double operator[](int i) const { return e[i]; }
    __host__ __device__ double &operator[](int i) { return e[i]; }

    __host__ __device__ vec3 &operator+=(const vec3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3 &operator*=(const vec3 &v)
    {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    __host__ __device__ vec3 &operator*=(double t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3 &operator/=(double t)
    {
        return *this *= 1 / t;
    }

    __host__ __device__ double lenght() const
    {
        return std::sqrt(lenght_squared());
    }

    __host__ __device__ double lenght_squared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __host__ __device__ bool near_zero() const
    {
        auto s = 1e-8;
        return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
    }

    // These random functions are now purely host-only, guarded by #ifndef __CUDACC__
#ifndef __CUDACC__
    static vec3 random()
    {
        return vec3(random_double(), random_double(), random_double());
    }

    static vec3 random(double min, double max)
    {
        return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
    }
#endif
};

// Alias
using point3 = vec3;

// Utilities
inline std::ostream &operator<<(std::ostream &out, const vec3 &v)
{
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3 &v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, double t)
{
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3 &v, double t)
{
    return (1 / t) * v;
}

__host__ __device__ inline double dot(const vec3 &u, const vec3 &v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3 &v)
{
    return v / v.lenght();
}

// These random vector functions are now purely host-only.
#ifndef __CUDACC__
inline vec3 random_unit_vector()
{
    while (true)
    {
        auto p = vec3::random(-1, 1);
        auto lensq = p.lenght_squared();
        if (1e-160 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
}

inline vec3 random_on_hemisphere(const vec3 &normal)
{
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0)
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}
#endif

__host__ __device__ inline vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - 2 * dot(v, n) * n;
}

// Add these for element-wise fmin/fmax for vec3
__host__ __device__ inline vec3 fmin_vec3(const vec3 &v1, const vec3 &v2)
{
    return vec3(fmin(v1.x(), v2.x()), fmin(v1.y(), v2.y()), fmin(v1.z(), v2.z()));
}

__host__ __device__ inline vec3 fmax_vec3(const vec3 &v1, const vec3 &v2)
{
    return vec3(fmax(v1.x(), v2.x()), fmax(v1.y(), v2.y()), fmax(v1.z(), v2.z()));
}

#endif