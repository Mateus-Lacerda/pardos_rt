#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "rtweekend.h"
#include "material.h"
#include "vec3.h"
#include <fstream>
#include <omp.h>

class camera : public movable
{
public:
    double aspect_ratio = 1.0;
    int image_width = 100;
    int samples_per_pixel = 10;
    int max_depth = 10;
    point3 center = point3(0, 0, 0);  // Agora público
    point3 lookat = point3(0, 0, -1); // Novo: ponto para onde a câmera olha

    // Novo: método para alterar o ponto para onde a câmera olha
    void look_at(const point3 &target)
    {
        lookat = target;
    }

    // Novo: método para rotacionar a direção da câmera em torno do eixo Y (esquerda/direita)
    void rotate_yaw(double angle_rad)
    {
        vec3 direction = lookat - center;
        double cos_a = cos(angle_rad);
        double sin_a = sin(angle_rad);
        double x = direction.x() * cos_a - direction.z() * sin_a;
        double z = direction.x() * sin_a + direction.z() * cos_a;
        lookat = center + vec3(x, direction.y(), z);
    }

    void render(const hittable &world, const std::string &filename)
    {
        initialize();

        std::ofstream out(filename, std::ios::out | std::ios::trunc);
        if (!out)
        {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        out << "P3\n"
            << image_width << ' ' << image_height << "\n255\n";

        for (int j = 0; j < image_height; j++)
        {
            std::clog << "\nScanlines remaining: " << (image_height - j) << ' ' << std::flush;
            for (int i = 0; i < image_width; i++)
            {
                // Initial white pixel
                color pixel_color(0, 0, 0);
                for (int sample = 0; sample < samples_per_pixel; sample++)
                {
                    // Ray at position (i, j)
                    // Could this be before the loop?
                    // No because get_ray introduces randomness for the antialiasing
                    ray r = get_ray(i, j);
                    // Increment the pixel_color
                    pixel_color += ray_color(r, max_depth, world);
                }
                // auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
                // auto ray_direction = pixel_center - center; // The - op does nothing
                //
                // // DEBUG
                // // std::clog << "\nPixel Center: " << pixel_center << ' ' << std::flush;
                // // std::clog << "\nRay Direction: " << ray_direction << ' ' << std::flush;
                //
                // ray r(center, ray_direction);
                //
                // color pixel_color = ray_color(r, world);

                write_color(out, pixel_samples_scale * pixel_color);
            }
        }

        std::clog << "\rDone                           \n";
        out.close();
    }
    color get_pixel_color(int i, int j, const hittable &world)
    {
        color pixel_color(0, 0, 0);
#pragma omp parallel
        {
            color local_color(0, 0, 0);
#pragma omp for nowait
            for (int sample = 0; sample < samples_per_pixel; sample++)
            {
                ray r = get_ray(i, j);
                local_color += ray_color(r, max_depth, world);
            }
#pragma omp critical
            {
                pixel_color += local_color;
            }
        }
        return pixel_samples_scale * pixel_color;
    }
    void initialize()
    {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        // Novo: direção da câmera
        vec3 forward = unit_vector(lookat - center);
        vec3 up = vec3(0, 1, 0);
        vec3 right = unit_vector(cross(forward, up));
        up = cross(right, forward);

        auto focal_length = 1.0;
        auto viewport_height = 2.0;
        auto viewport_width = viewport_height * (double(image_width) / image_height);

        auto viewport_u = viewport_width * right;
        auto viewport_v = -viewport_height * up;

        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        auto viewport_upper_left = center + forward * focal_length - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

    void move(char &movement) override
    {
        // Calcula vetores relativos à direção atual
        vec3 forward = unit_vector(lookat - center);
        vec3 up = vec3(0, 1, 0);
        vec3 right = unit_vector(cross(forward, up));
        up = cross(right, forward);
        double step = 0.01;
        switch (movement)
        {
        case 'w': // Frente
            center = center + step * forward;
            lookat = lookat + step * forward;
            break;
        case 's': // Trás
            center = center - step * forward;
            lookat = lookat - step * forward;
            break;
        case 'a': // Esquerda
            center = center - step * right;
            lookat = lookat - step * right;
            break;
        case 'd': // Direita
            center = center + step * right;
            lookat = lookat + step * right;
            break;
        case 'j': // Cima
            center = center + step * up;
            lookat = lookat + step * up;
            break;
        case 'k': // Baixo
            center = center - step * up;
            lookat = lookat - step * up;
            break;
        case 'q': // Girar para a esquerda
            rotate_yaw(-0.05);
            break;
        case 'e': // Girar para a direita
            rotate_yaw(0.05);
            break;
        }
    }

private:
    int image_height;
    double pixel_samples_scale;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;

    ray get_ray(int i, int j) const
    {
        // Offset is a randomized vec3, at z = 0
        auto offset = sample_square();
        auto pixel_sample = pixel00_loc // Get the 0th pixel`s pos
                                        // Walk offset.x pixels
                            + ((i + offset.x()) * pixel_delta_u)
                            // Walk offset.y pixels
                            + ((j + offset.y()) * pixel_delta_v);
        // Ray origin at the center
        auto ray_origin = center;
        // Vector op to get the ray direction to the sample
        auto ray_direction = pixel_sample - ray_origin;
        // Returns a ray from the origin to the sampled direction
        return ray(ray_origin, ray_direction);
    }

    vec3 sample_square() const
    {
        return vec3(random_double() - 0.5, random_double() - 0.5, 0);
    }

    color ray_color(const ray &r, int depth, const hittable &world) const
    {
        if (depth <= 0)
            // If light can't hit anything, its just light
            return color(0, 0, 0);

        hit_record rec;

        if (world.hit(r, interval(0.001, infinity), rec))
        {
            // return 0.5 * (rec.normal + color(1,1,1));
            // Random ray diffusing
            // vec3 direction = random_on_hemisphere(rec.normal);
            // Lambertian distribution diffusing
            // vec3 direction = rec.normal + random_unit_vector();
            // return 0.5 * ray_color(ray(rec.p, direction), depth-1, world);
            // Material specific diffusing
            ray scattered;
            color attenuation;
            if (rec.mat->scatter(r, rec, attenuation, scattered))
                return attenuation * ray_color(scattered, depth - 1, world);
            return color(0, 0, 0);
        }

        // Lerp
        // blendedValue=(1−a)⋅startValue+a⋅endValue
        // First, create a unit vetor in the ray direction
        vec3 unit_direction = unit_vector(r.direction());
        // Based on the y coordinaate
        auto a = 0.5 * (unit_direction.y() + 1.0);
        // std::clog << "\na = " << a << ' ' << std::flush;
        return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
    }
};

#endif // !CAMERA_H
