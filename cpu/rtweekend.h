#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <thread>

// C++ Std Usings

using std::make_shared;
using std::shared_ptr;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees)
{
    return degrees * pi / 180.0;
}

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

// Common Headers

#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"
#include "movable.h"

#endif
