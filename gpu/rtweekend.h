#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <thread>

// Include CUDA utilities for device-side random functions if compiling with nvcc
#ifdef __CUDACC__
#include "cuda_utils.cuh"
#endif

// C++ Std Usings

// shared_ptr is generally for host-side. For device, we'll use raw pointers or simpler structs.
using std::make_shared;
using std::shared_ptr;

// Constants
// Use 'constexpr' for device compatibility where possible
#ifdef __CUDACC__
// For CUDA device code, ensure constants are accessible
#define CONST_VAR __device__ const
#else
#define CONST_VAR const
#endif

CONST_VAR double infinity = std::numeric_limits<double>::infinity();
CONST_VAR double pi = 3.1415926535897932385;

// Utility Functions

__host__ __device__ inline double degrees_to_radians(double degrees)
{
    return degrees * pi / 180.0;
}

// These random functions use thread_local std::mt19937, which is for CPU.
// For CUDA, we'll need a separate mechanism (handled in cuda_utils.cuh).
// Therefore, these are compiled only for the host.
#ifndef __CUDACC__
inline double random_double()
{
    thread_local static std::mt19937 generator(std::random_device{}() + std::hash<std::thread::id>{}(std::this_thread::get_id()));
    thread_local static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}

inline double random_double(double min, double max)
{
    thread_local static std::mt19937 generator(std::random_device{}() + std::hash<std::thread::id>{}(std::this_thread::get_id()));
    std::uniform_real_distribution<double> distribution(min, max);
    return distribution(generator);
}
#endif // !__CUDACC__

// Common Headers (ensure these are placed after definitions if they depend on them)

#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"
#include "movable.h" // Assuming this is a necessary header

#endif