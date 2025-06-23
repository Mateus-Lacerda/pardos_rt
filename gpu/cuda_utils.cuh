#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "vec3.h"          // For vec3, point3
#include "curand_kernel.h" // For curandState

// Device-side random number generation using CURAND
__device__ inline double random_double_device(curandState *state)
{
    return curand_uniform(state);
}

__device__ inline double random_double_device(double min, double max, curandState *state)
{
    return min + (max - min) * curand_uniform(state);
}

// Device-side random vector generation
__device__ inline vec3 random_vec3_device(curandState *state)
{
    return vec3(random_double_device(state), random_double_device(state), random_double_device(state));
}

__device__ inline vec3 random_vec3_device(double min, double max, curandState *state)
{
    return vec3(random_double_device(min, max, state), random_double_device(min, max, state), random_double_device(min, max, state));
}

__device__ inline vec3 random_unit_vector_device(curandState *state)
{
    while (true)
    {
        auto p = random_vec3_device(-1, 1, state);
        auto lensq = p.lenght_squared();
        if (1e-160 < lensq && lensq <= 1) // Ensure length > 0
            return p / sqrt(lensq);
    }
}

__device__ inline vec3 random_on_hemisphere_device(const vec3 &normal, curandState *state)
{
    vec3 on_unit_sphere = random_unit_vector_device(state);
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

#endif // CUDA_UTILS_CUH