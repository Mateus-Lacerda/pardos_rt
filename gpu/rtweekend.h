#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

#ifdef __CUDACC__
#include <curand_kernel.h> // Para o curandState
#endif

// Usings
using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions
inline double degrees_to_radians(double degrees)
{
    return degrees * pi / 180.0;
}

// Common Headers
#include "vec3.h"
#include "color.h"
#include "ray.h"

#ifdef __CUDACC__
#include "cuda_utils.cuh"
#endif

#endif