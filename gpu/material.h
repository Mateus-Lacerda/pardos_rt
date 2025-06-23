#ifndef MATERIAL_H
#define MATERIAL_H

// **CRITICAL FIX: Remove direct include of hittable.h to break circular dependency**
// #include "hittable.h" // REMOVED THIS LINE

#include "rtweekend.h"
#include "vec3.h"
// **CRITICAL FIX: Wrap curand_kernel.h include with #ifdef __CUDACC__**
// This ensures it's only included when compiled by NVCC, not by g++
#ifdef __CUDACC__
#include "curand_kernel.h" // Needed for curandState in __device__ scatter_gpu
#endif

// Forward declarations for types used in __device__ scatter_gpu prototype
// These are needed because hittable.h is no longer directly included here.
struct hit_record; // Forward declaration for hit_record
struct ray;        // Forward declaration for ray

// Forward declaration for random_double from rtweekend.h for host-only calls
#ifndef __CUDACC__
inline double random_double(); // Ensure this is available for host-side scatter
#endif

// GPU-compatible material structs (POD types)
// No __host__ __device__ on the struct declarations themselves.
struct GPULambertian
{
    color albedo;
};

struct GPUMetal
{
    color albedo;
    double fuzz;
};

struct GPUDielectric
{
    double ir; // Index of Refraction
};

// Unified Material struct for GPU - DEFINED HERE AFTER ITS MEMBERS ARE DEFINED
struct GPUMaterial
{
    int type; // 0: Lambertian, 1: Metal, 2: Dielectric
    // Using union to save memory, as only one type will be active
    union
    {
        GPULambertian lambertian_data;
        GPUMetal metal_data;
        GPUDielectric dielectric_data;
    };
    // Explicit default constructor for GPUMaterial to prevent hit_record errors
    // Since it has a union, the compiler might delete default/copy/move ops.
    __host__ __device__ GPUMaterial() : type(0) {} // Initialize type to a default, e.g., Lambertian
};

// __device__ scatter function that replaces polymorphic scatter - defined in .cu file
// It must accept curandState* for device-side randomness.
// Now uses forward-declared hit_record and ray.
__device__ bool scatter_gpu(
    const GPUMaterial &mat, const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered, curandState *rand_state);

// Host-side polymorphic material classes (remain the same for scene setup)
// *** WRAPPED IN #ifndef __CUDACC__ TO PREVENT DEVICE COMPILATION ***
#ifndef __CUDACC__
// class material and its derived classes (lambertian, metal, dielectric)
// are defined here. Their methods call host-only random functions.

class material
{
public:
    virtual ~material() = default;

    virtual bool scatter(
        const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const
    {
        return false;
    }
};

class lambertian : public material
{
public:
    lambertian(const color &a) : albedo(a) {}

    bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
        const override
    {
        auto scatter_direction = rec.normal + random_unit_vector(); // Calls host-only random_unit_vector()

        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

private:
    color albedo;
};

class metal : public material
{
public:
    metal(const color &albedo, double f) : albedo(albedo), fuzz(f < 1 ? f : 1) {}

    bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
        const override
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_unit_vector()); // Calls host-only random_unit_vector()
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

private:
    color albedo;
    double fuzz;
};

class dielectric : public material
{
public:
    dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override
    {
        attenuation = color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double()) // Calls host-only random_double()
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

private:
    double ir; // Index of Refraction

    static double reflectance(double cosine, double ref_idx)
    {
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }

    vec3 refract(const vec3 &uv, const vec3 &n, double etai_over_etat) const
    {
        auto cos_theta = fmin(dot(-uv, n), 1.0);
        vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
        vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.lenght_squared())) * n;
        return r_out_perp + r_out_parallel;
    }
};
#endif // !__CUDACC__ for host-side materials

#endif