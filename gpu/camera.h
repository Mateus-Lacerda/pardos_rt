#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h" // For hit_record, hittable (though ray_color is gone)
#include "rtweekend.h"
#include "material.h" // For material (though scatter is gone)
#include "vec3.h"
#include "movable.h" // For movable base class
// #include <fstream> // Not needed if render is removed
// #include <omp.h>   // Not needed if OpenMP pragmas are removed

// GPU-compatible struct for camera parameters.
// This is a plain data struct that can be copied to the GPU.
struct GPUCamera
{
    point3 center;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 u, v, w;          // Camera basis vectors
    double defocus_disk_u; // Length of defocus_disk_u
    double defocus_disk_v; // Length of defocus_disk_v
    double time0;
    double time1;
};

class camera : public movable
{
public:
    double aspect_ratio = 1.0;
    int image_width = 100;
    int samples_per_pixel = 10;
    int max_depth = 10;
    point3 center = point3(0, 0, 0);  // Now public
    point3 lookat = point3(0, 0, -1); // New: point camera is looking at
    vec3 vup = vec3(0, 1, 0);         // Camera's "up" direction
    double vfov = 20;                 // Vertical field of view in degrees

    // Defocus parameters
    double defocus_angle = 0; // Variation angle of rays through each pixel
    double focus_dist = 10;   // Distance from camera lookfrom point to plane of perfect focus

    // Time for motion blur
    double time0 = 0.0; // Start of shutter open time
    double time1 = 1.0; // End of shutter open time

    // Public members to be passed to GPUCamera
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 u, v, w;          // Camera basis vectors for defocus disk calculations
    double defocus_disk_u; // Horizontal defocus disk radius
    double defocus_disk_v; // Vertical defocus disk radius

    // New: method to change the point the camera looks at
    void look_at(const point3 &target)
    {
        lookat = target;
    }

    // New: method to rotate the camera direction around the Y-axis (left/right)
    void rotate_yaw(double angle_rad)
    {
        vec3 direction = lookat - center;
        double cos_a = cos(angle_rad);
        double sin_a = sin(angle_rad);
        double x = direction.x() * cos_a - direction.z() * sin_a;
        double z = direction.x() * sin_a + direction.z() * cos_a;
        lookat = center + vec3(x, direction.y(), z);
    }

    // `render` method is removed as rendering is now handled by SDLRenderer using CUDA.
    // `get_pixel_color` is removed as its logic is now in the CUDA kernel.

    void initialize()
    {
        // Calculate image height
        int image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        // Camera basis vectors
        w = unit_vector(center - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Viewport dimensions
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2 * h * focus_dist; // Viewport height at focus distance
        auto viewport_width = viewport_height * (double(image_width) / image_height);

        // Calculate vectors across the horizontal and vertical viewport edges
        vec3 viewport_u = viewport_width * u;     // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * (-v); // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper-left pixel
        auto viewport_upper_left = center - (w * focus_dist) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors
        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u.lenght() * defocus_radius; // Use length to scale from unit vector u
        defocus_disk_v = v.lenght() * defocus_radius; // Use length to scale from unit vector v
    }

    void move(char &movement) override
    {
        // Calculate vectors relative to the current direction
        vec3 forward = unit_vector(lookat - center);
        vec3 up_dir = vup; // Use vup as the base for up movement
        vec3 right_dir = unit_vector(cross(forward, up_dir));

        double step = 0.01;
        switch (movement)
        {
        case 'w': // Forward
            center = center + step * forward;
            lookat = lookat + step * forward;
            break;
        case 's': // Back
            center = center - step * forward;
            lookat = lookat - step * forward;
            break;
        case 'a': // Left
            center = center - step * right_dir;
            lookat = lookat - step * right_dir;
            break;
        case 'd': // Right
            center = center + step * right_dir;
            lookat = lookat + step * right_dir;
            break;
        case 'j': // Up
            center = center + step * up_dir;
            lookat = lookat + step * up_dir;
            break;
        case 'k': // Down
            center = center - step * up_dir;
            lookat = lookat - step * up_dir;
            break;
        case 'q': // Rotate left (yaw)
            rotate_yaw(-0.05);
            break;
        case 'e': // Rotate right (yaw)
            rotate_yaw(0.05);
            break;
        }
    }

private:
    // image_height is now a local variable in initialize() if not needed elsewhere
    // double pixel_samples_scale; // Not needed on host anymore
};

#endif // !CAMERA_H