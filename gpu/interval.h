#ifndef INTERVAL_H
#define INTERVAL_H
#include "rtweekend.h"

class interval
{
public:
    double min, max;

    __host__ __device__ interval() : min(+infinity), max(-infinity) {} // Default interval empty
    __host__ __device__ interval(double min, double max) : min(min), max(max) {}
    __host__ __device__ double size() const { return max - min; }
    __host__ __device__ bool contains(double x) const { return min <= x && x <= max; }
    __host__ __device__ bool surrounds(double x) const { return min < x && x < max; }
    __host__ __device__ double clamp(double x) const
    {
        if (x < min)
            return min;
        if (x > max)
            return max;
        return x;
    }
    // Constants for host and device - REMOVED __host__ __device__ from declarations inside class
    static const interval empty, universe;
};

// Define static members for both host and device - REMOVED __host__ __device__ from definitions here
// These will now be simple inline const objects. nvcc handles their constant memory/copying.
inline const interval interval::empty = interval(+infinity, -infinity);
inline const interval interval::universe = interval(-infinity, +infinity);

#endif // !INTERVAL_H