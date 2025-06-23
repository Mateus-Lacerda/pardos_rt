#ifndef MATERIAL_H
#define MATERIAL_H

#include "rtweekend.h" // Includes vec3, color, etc.

// --- GPU-compatible Material Structs ---
// These are plain data structs that can be copied to the GPU.
// They correspond to the host-side material classes.

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

// A union to hold different material data, discriminated by 'type'.
// This is how we achieve a form of polymorphism on the GPU.
struct GPUMaterial
{
    int type; // 0: Lambertian, 1: Metal, 2: Dielectric
    union
    {
        GPULambertian lambertian_data;
        GPUMetal metal_data;
        GPUDielectric dielectric_data;
    };

    // Default constructor to satisfy initialization in hit_record
    __host__ __device__ GPUMaterial() : type(0)
    {
        lambertian_data.albedo = color(0, 0, 0);
    }
};

class material
{
public:
    virtual ~material() = default;
};

class lambertian : public material
{
public:
    lambertian(const color &a) : albedo(a) {}

    // Public for access from sdl_renderer.cc
    color albedo;
};

class metal : public material
{
public:
    metal(const color &a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    // Public for access from sdl_renderer.cc
    color albedo;
    double fuzz;
};

class dielectric : public material
{
public:
    dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    // Public for access from sdl_renderer.cc
    double ir;
};

#endif