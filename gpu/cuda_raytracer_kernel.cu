// **CRITICAL: Correct Include Order**
// material.h MUST be included before hittable.h because hittable.h uses GPUMaterial.
// Also, camera.h before material.h if material.h uses camera types (e.g. point3/vec3).
#include "rtweekend.h"
#include "camera.h"        // For Camera class parameters
#include "material.h"      // <--- Include material.h FIRST, as it defines GPUMaterial
#include "hittable.h"      // <--- Include hittable.h SECOND, as it uses GPUMaterial
#include "cuda_utils.cuh"  // For device-side randoms
#include "curand_kernel.h" // For curandState

// --- Device-side Camera Parameters Structure ---
// REMOVED __attribute__((device)) (and implicit __device__ keyword) as it's for functions, not struct declarations.
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

// External declaration for the CUDA kernel (to be linked from sdl_renderer.cc)
extern "C" __global__ void render_kernel(
    color *d_pixels,
    int image_width, int image_height,
    const GPUCamera d_camera_params,
    const GPUSphere *d_spheres, int num_spheres,
    const GPUPlane *d_planes, int num_planes,
    const GPUBox *d_boxes, int num_boxes,
    const GPUCylinder *d_cylinders, int num_cylinders,
    curandState *d_rand_states,
    int samples_per_pixel, int max_depth);

// --- Device-side Hittable Functions ---

// __device__ hit function for GPUSphere
__device__ bool hit_sphere_gpu(
    const GPUSphere &s, const ray &r, interval ray_t, hit_record &rec, double time)
{
    point3 center = s.is_moving ? s.center + time * s.center_vec : s.center;
    vec3 oc = r.origin() - center;
    auto a = r.direction().lenght_squared();
    auto h = dot(oc, r.direction());
    auto c = oc.lenght_squared() - s.radius * s.radius;

    auto discriminant = h * h - a * c;
    if (discriminant < 0)
        return false;

    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-h - sqrtd) / a;
    if (!ray_t.surrounds(root))
    {
        root = (-h + sqrtd) / a;
        if (!ray_t.surrounds(root))
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / s.radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat = s.material_data; // Assign the GPU material data
    return true;
}

// __device__ hit function for GPUPlane
__device__ bool hit_plane_gpu(
    const GPUPlane &p, const ray &r, interval ray_t, hit_record &rec)
{
    vec3 normal = p.normal;
    double denominator = dot(normal, r.direction());

    // Ray is parallel to the plane or nearly so
    if (fabs(denominator) < 1e-8)
    {
        return false;
    }

    double t = dot((p.point - r.origin()), normal) / denominator;

    if (!ray_t.surrounds(t))
    {
        return false;
    }

    rec.t = t;
    rec.p = r.at(rec.t);
    rec.set_face_normal(r, normal);
    rec.mat = p.material_data;
    return true;
}

// __device__ hit function for GPUBox (AABB)
__device__ bool hit_box_gpu(
    const GPUBox &b, const ray &r, interval ray_t, hit_record &rec)
{
    vec3 inv_dir = vec3(1.0 / r.direction().x(), 1.0 / r.direction().y(), 1.0 / r.direction().z());
    vec3 t0s = (b.box_min - r.origin()) * inv_dir;
    vec3 t1s = (b.box_max - r.origin()) * inv_dir;

    vec3 tmin = fmin_vec3(t0s, t1s);
    vec3 tmax = fmax_vec3(t0s, t1s);

    double t_enter = fmax(fmax(tmin.x(), tmin.y()), tmin.z());
    double t_exit = fmin(fmin(tmax.x(), tmax.y()), tmax.z());

    if (t_exit < t_enter || t_exit <= ray_t.min || t_enter >= ray_t.max)
    {
        return false;
    }

    double hit_t = t_enter;
    if (!ray_t.surrounds(hit_t))
    {
        hit_t = t_exit;
        if (!ray_t.surrounds(hit_t))
        {
            return false;
        }
    }

    rec.t = hit_t;
    rec.p = r.at(rec.t);

    vec3 outward_normal;
    const double epsilon = 1e-6;
    if (fabs(rec.p.x() - b.box_min.x()) < epsilon)
        outward_normal = vec3(-1, 0, 0);
    else if (fabs(rec.p.x() - b.box_max.x()) < epsilon)
        outward_normal = vec3(1, 0, 0);
    else if (fabs(rec.p.y() - b.box_min.y()) < epsilon)
        outward_normal = vec3(0, -1, 0);
    else if (fabs(rec.p.y() - b.box_max.y()) < epsilon)
        outward_normal = vec3(0, 1, 0);
    else if (fabs(rec.p.z() - b.box_min.z()) < epsilon)
        outward_normal = vec3(0, 0, -1);
    else if (fabs(rec.p.z() - b.box_max.z()) < epsilon)
        outward_normal = vec3(0, 0, 1);
    else
        outward_normal = vec3(0, 0, 0);

    rec.set_face_normal(r, outward_normal);
    rec.mat = b.material_data;
    return true;
}

// __device__ hit function for GPUCylinder (infinite cylinder along Y-axis, then clipped)
__device__ bool hit_cylinder_gpu(
    const GPUCylinder &cyl, const ray &r, interval ray_t, hit_record &rec)
{
    vec3 oc = r.origin() - cyl.center;
    double a = r.direction().x() * r.direction().x() + r.direction().z() * r.direction().z();
    double half_b = oc.x() * r.direction().x() + oc.z() * r.direction().z();
    double c = oc.x() * oc.x() + oc.z() * oc.z() - cyl.radius * cyl.radius;

    double discriminant = half_b * half_b - a * c;
    if (discriminant < 0)
        return false;

    double sqrtd = sqrt(discriminant);

    double root = (-half_b - sqrtd) / a;
    if (!ray_t.surrounds(root))
    {
        root = (-half_b + sqrtd) / a;
        if (!ray_t.surrounds(root))
            return false;
    }

    point3 hit_point = r.at(root);
    double y_local = hit_point.y() - cyl.center.y();
    if (y_local < -cyl.half_height || y_local > cyl.half_height)
    {
        return false;
    }

    rec.t = root;
    rec.p = hit_point;
    vec3 outward_normal = unit_vector(vec3(rec.p.x() - cyl.center.x(), 0, rec.p.z() - cyl.center.z()));
    rec.set_face_normal(r, outward_normal);
    rec.mat = cyl.material_data;
    return true;
}

// --- Device-side Material Scattering Function ---

__device__ bool scatter_gpu(
    const GPUMaterial &mat, const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered, curandState *rand_state)
{
    switch (mat.type)
    {
    case 0:
    { // Lambertian
        auto scatter_direction = rec.normal + random_unit_vector_device(rand_state);

        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = mat.lambertian_data.albedo;
        return true;
    }
    case 1:
    { // Metal
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + mat.metal_data.fuzz * random_unit_vector_device(rand_state));
        attenuation = mat.metal_data.albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
    case 2:
    { // Dielectric
        attenuation = color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0 / mat.dielectric_data.ir) : mat.dielectric_data.ir;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        auto reflectance_schlick = [] __device__(double cosine, double ref_idx)
        {
            auto r0 = (1 - ref_idx) / (1 + ref_idx);
            r0 = r0 * r0;
            return r0 + (1 - r0) * pow((1 - cosine), 5);
        };

        if (cannot_refract || reflectance_schlick(cos_theta, refraction_ratio) > random_double_device(rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
        {
            auto refract_vec = [] __device__(const vec3 &uv, const vec3 &n, double etai_over_etat)
            {
                auto cos_theta_refract = fmin(dot(-uv, n), 1.0);
                vec3 r_out_perp = etai_over_etat * (uv + cos_theta_refract * n);
                vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.lenght_squared())) * n;
                return r_out_perp + r_out_parallel;
            };
            direction = refract_vec(unit_direction, rec.normal, refraction_ratio);
        }

        scattered = ray(rec.p, direction);
        return true;
    }
    default:
        return false;
    }
}

// --- Device-side Ray Color Calculation ---
__device__ color ray_color_gpu(
    const ray &r, int max_depth, curandState *rand_state,
    const GPUSphere *d_spheres, int num_spheres,
    const GPUPlane *d_planes, int num_planes,
    const GPUBox *d_boxes, int num_boxes,
    const GPUCylinder *d_cylinders, int num_cylinders,
    double time0, double time1)
{
    ray current_r = r;
    color accumulated_color = color(1.0, 1.0, 1.0);

    for (int depth = 0; depth < max_depth; ++depth)
    {
        hit_record rec;
        interval ray_t = interval(0.001, infinity);
        bool hit_anything = false;
        double closest_so_far = infinity;

        for (int i = 0; i < num_spheres; ++i)
        {
            hit_record temp_rec;
            if (hit_sphere_gpu(d_spheres[i], current_r, interval(ray_t.min, closest_so_far), temp_rec, random_double_device(time0, time1, rand_state)))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        for (int i = 0; i < num_planes; ++i)
        {
            hit_record temp_rec;
            if (hit_plane_gpu(d_planes[i], current_r, interval(ray_t.min, closest_so_far), temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        for (int i = 0; i < num_boxes; ++i)
        {
            hit_record temp_rec;
            if (hit_box_gpu(d_boxes[i], current_r, interval(ray_t.min, closest_so_far), temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        for (int i = 0; i < num_cylinders; ++i)
        {
            hit_record temp_rec;
            if (hit_cylinder_gpu(d_cylinders[i], current_r, interval(ray_t.min, closest_so_far), temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        if (hit_anything)
        {
            ray scattered;
            color attenuation;
            if (scatter_gpu(rec.mat, current_r, rec, attenuation, scattered, rand_state))
            {
                accumulated_color = accumulated_color * attenuation;
                current_r = scattered;
            }
            else
            {
                return color(0, 0, 0);
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(current_r.direction());
            auto a = 0.5 * (unit_direction.y() + 1.0);
            return accumulated_color * ((1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0));
        }
    }
    return color(0, 0, 0);
}

// --- Main CUDA Rendering Kernel ---
extern "C" __global__ void render_kernel(
    color *d_pixels,
    int image_width, int image_height,
    const GPUCamera d_camera_params,
    const GPUSphere *d_spheres, int num_spheres,
    const GPUPlane *d_planes, int num_planes,
    const GPUBox *d_boxes, int num_boxes,
    const GPUCylinder *d_cylinders, int num_cylinders,
    curandState *d_rand_states,
    int samples_per_pixel, int max_depth)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= image_width || j >= image_height)
        return;

    curandState *rand_state = &d_rand_states[j * image_width + i];

    // Initialize CURAND state for this thread using a unique seed
    curand_init(1234 + (j * image_width + i), 0, 0, rand_state);

    color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; ++s)
    {
        auto r_double1_pixel = random_double_device(rand_state);
        auto r_double2_pixel = random_double_device(rand_state);

        auto pixel_center = d_camera_params.pixel00_loc + (i * d_camera_params.pixel_delta_u) + (j * d_camera_params.pixel_delta_v);
        auto pixel_sample = pixel_center + (r_double1_pixel * d_camera_params.pixel_delta_u) + (r_double2_pixel * d_camera_params.pixel_delta_v);

        point3 ray_origin = d_camera_params.center;
        if (d_camera_params.defocus_disk_u > 1e-8 || d_camera_params.defocus_disk_v > 1e-8)
        {
            vec3 p_disk_unit = random_unit_vector_device(rand_state);
            double random_radius = random_double_device(rand_state);
            vec3 p_disk = p_disk_unit * random_radius;

            ray_origin = d_camera_params.center + (p_disk.x() * d_camera_params.u * d_camera_params.defocus_disk_u) + (p_disk.y() * d_camera_params.v * d_camera_params.defocus_disk_v);
        }

        double ray_time = random_double_device(d_camera_params.time0, d_camera_params.time1, rand_state);
        ray r_single_sample(ray_origin, pixel_sample - ray_origin);

        pixel_color += ray_color_gpu(r_single_sample, max_depth, rand_state,
                                     d_spheres, num_spheres,
                                     d_planes, num_planes,
                                     d_boxes, num_boxes,
                                     d_cylinders, num_cylinders,
                                     d_camera_params.time0, d_camera_params.time1);
    }
    pixel_color /= samples_per_pixel;

    pixel_color.e[0] = fmax(0.0, fmin(1.0, pixel_color.x()));
    pixel_color.e[1] = fmax(0.0, fmin(1.0, pixel_color.y()));
    pixel_color.e[2] = fmax(0.0, fmin(1.0, pixel_color.z()));

    d_pixels[j * image_width + i] = pixel_color;
}