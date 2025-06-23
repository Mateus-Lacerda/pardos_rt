#include "sdl_renderer.h"
#include <SDL2/SDL.h>
#include <vector>
#include <algorithm>
#include <iostream> // For std::cerr and printing CUDA errors
#include <numeric>  // For std::iota (though not strictly used here for CURAND init)

// Include concrete primitive headers for dynamic_pointer_cast
// These are needed to correctly identify the type of hittable object
// from the polymorphic hittable_map for conversion to GPU structs.
#include "sphere.h"
#include "plane.h"
#include "box.h"
#include "cylinder.h"
#include "material.h"
#include "hittable.h"
#include "hittable_list.h"
#include "vec3.h" // For color and vec3 definitions

// The camera class and the GPUCamera struct are now included from camera.h
#include "camera.h"

// Extern declaration for the CUDA kernel
// This tells the compiler that `render_kernel` is defined elsewhere (in cuda_raytracer_kernel.cu)
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

// Helper for CUDA error checking
// This macro simplifies checking for CUDA errors after each API call.
#define CudaCheckError()                                                                               \
    {                                                                                                  \
        cudaError_t err = cudaGetLastError();                                                          \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    }

// Constructor: Initializes SDL and allocates initial GPU memory
SDLRenderer::SDLRenderer(int img_w, int img_h, int win_w, int win_h)
    : width(img_w), height(img_h), window_width(win_w), window_height(win_h),
      pixels_host(img_w * img_h, 0), pixels_gpu_result(img_w * img_h), // Initialize the new buffer
      d_pixels(nullptr), d_rand_states(nullptr),
      d_spheres(nullptr), d_planes(nullptr), d_boxes(nullptr), d_cylinders(nullptr),
      // Initialize counts to zero
      num_spheres(0), num_planes(0), num_boxes(0), num_cylinders(0)
{
    // Initialize SDL
    SDL_Init(SDL_INIT_VIDEO);
    window = SDL_CreateWindow("Raytracer (CUDA)", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, window_width, window_height, 0);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height);

    // Allocate device memory for pixels and CURAND states once
    // These buffers are persistent across renders, only scene data is re-allocated if scene changes.
    cudaMalloc(&d_pixels, width * height * sizeof(color));
    CudaCheckError();
    cudaMalloc(&d_rand_states, width * height * sizeof(curandState));
    CudaCheckError();

    // Initialize CURAND states on the device.
    // We launch a small kernel to initialize the random states for each pixel.
    // The seed can be fixed or based on time for different results.
    // Note: This 'init_curand' is a conceptual wrapper. The actual kernel
    // `init_curand_states_kernel` needs to be defined in `cuda_raytracer_kernel.cu`.
    // For simplicity, the main `render_kernel` will initialize `curandState` itself.
    // So, `init_curand()` as a separate method is actually not needed if `curand_init`
    // is called per thread within the `render_kernel`.
    // Leaving `init_curand` in .h but commenting out its body for now.
    // The `d_rand_states` is just memory for the states, `curand_init` populates it.
}

// Destructor: Frees all allocated resources
SDLRenderer::~SDLRenderer()
{
    free_gpu_scene(); // Free scene-specific device memory

    // Free persistent GPU memory
    if (d_pixels)
    {
        cudaFree(d_pixels);
        CudaCheckError();
    }
    if (d_rand_states)
    {
        cudaFree(d_rand_states);
        CudaCheckError();
    }

    // Destroy SDL resources
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

// Placeholder for init_curand - not strictly needed if curand_init is in the kernel
void SDLRenderer::init_curand()
{
    // No explicit host-side CURAND generator creation needed.
    // Each thread will initialize its own curandState in the kernel.
    // The `d_rand_states` memory is just where those states will live.
}

// Helper to prepare and copy scene data to the GPU
void SDLRenderer::setup_gpu_scene(const hittable_list &world)
{
    // First, free any previously allocated scene data to prevent memory leaks
    free_gpu_scene();

    // Host-side vectors to hold GPU-compatible structs
    std::vector<GPUSphere> host_spheres;
    std::vector<GPUPlane> host_planes;
    std::vector<GPUBox> host_boxes;
    std::vector<GPUCylinder> host_cylinders;

    // Iterate through the world's objects and convert them to GPU-compatible structs
    // With hittable_list, obj_ptr is now correctly a shared_ptr<hittable>.
    for (const auto &obj_ptr : world.objects)
    {
        // Use dynamic_pointer_cast to identify the concrete type of each hittable
        if (auto sphere_ptr = std::dynamic_pointer_cast<sphere>(obj_ptr)) // This now works correctly
        {
            GPUSphere gpu_s;
            gpu_s.center = sphere_ptr->center;
            gpu_s.radius = sphere_ptr->radius;
            gpu_s.is_moving = sphere_ptr->is_moving;
            gpu_s.center_vec = sphere_ptr->center_vec;

            // Convert material. This is where the material polymorphism is flattened.
            if (auto lam_mat = std::dynamic_pointer_cast<lambertian>(sphere_ptr->mat))
            {
                gpu_s.material_data.type = 0; // Lambertian
                gpu_s.material_data.lambertian_data.albedo = lam_mat->albedo;
            }
            else if (auto metal_mat = std::dynamic_pointer_cast<metal>(sphere_ptr->mat))
            {
                gpu_s.material_data.type = 1; // Metal
                gpu_s.material_data.metal_data.albedo = metal_mat->albedo;
                gpu_s.material_data.metal_data.fuzz = metal_mat->fuzz;
            }
            else if (auto die_mat = std::dynamic_pointer_cast<dielectric>(sphere_ptr->mat))
            {
                gpu_s.material_data.type = 2; // Dielectric
                gpu_s.material_data.dielectric_data.ir = die_mat->ir;
            }
            else
            {
                // Default to a black lambertian if material type is unknown
                gpu_s.material_data.type = 0;
                gpu_s.material_data.lambertian_data.albedo = color(0, 0, 0);
            }
            host_spheres.push_back(gpu_s);
        }
        else if (auto plane_ptr = std::dynamic_pointer_cast<plane>(obj_ptr))
        {
            GPUPlane gpu_p;
            gpu_p.point = plane_ptr->point;
            gpu_p.normal = plane_ptr->normal;

            if (auto lam_mat = std::dynamic_pointer_cast<lambertian>(plane_ptr->mat))
            {
                gpu_p.material_data.type = 0; // Lambertian
                gpu_p.material_data.lambertian_data.albedo = lam_mat->albedo;
            }
            else if (auto metal_mat = std::dynamic_pointer_cast<metal>(plane_ptr->mat))
            {
                gpu_p.material_data.type = 1; // Metal
                gpu_p.material_data.metal_data.albedo = metal_mat->albedo;
                gpu_p.material_data.metal_data.fuzz = metal_mat->fuzz;
            }
            else if (auto die_mat = std::dynamic_pointer_cast<dielectric>(plane_ptr->mat))
            {
                gpu_p.material_data.type = 2; // Dielectric
                gpu_p.material_data.dielectric_data.ir = die_mat->ir;
            }
            else
            {
                gpu_p.material_data.type = 0;
                gpu_p.material_data.lambertian_data.albedo = color(0, 0, 0);
            }
            host_planes.push_back(gpu_p);
        }
        else if (auto box_ptr = std::dynamic_pointer_cast<box>(obj_ptr))
        {
            GPUBox gpu_b;
            gpu_b.box_min = box_ptr->box_min;
            gpu_b.box_max = box_ptr->box_max;

            if (auto lam_mat = std::dynamic_pointer_cast<lambertian>(box_ptr->mat))
            {
                gpu_b.material_data.type = 0; // Lambertian
                gpu_b.material_data.lambertian_data.albedo = lam_mat->albedo;
            }
            else if (auto metal_mat = std::dynamic_pointer_cast<metal>(box_ptr->mat))
            {
                gpu_b.material_data.type = 1; // Metal
                gpu_b.material_data.metal_data.albedo = metal_mat->albedo;
                gpu_b.material_data.metal_data.fuzz = metal_mat->fuzz;
            }
            else if (auto die_mat = std::dynamic_pointer_cast<dielectric>(box_ptr->mat))
            {
                gpu_b.material_data.type = 2; // Dielectric
                gpu_b.material_data.dielectric_data.ir = die_mat->ir;
            }
            else
            {
                gpu_b.material_data.type = 0;
                gpu_b.material_data.lambertian_data.albedo = color(0, 0, 0);
            }
            host_boxes.push_back(gpu_b);
        }
        else if (auto cylinder_ptr = std::dynamic_pointer_cast<cylinder>(obj_ptr))
        {
            GPUCylinder gpu_c;
            gpu_c.center = cylinder_ptr->center;
            gpu_c.radius = cylinder_ptr->radius;
            gpu_c.half_height = cylinder_ptr->half_height;

            if (auto lam_mat = std::dynamic_pointer_cast<lambertian>(cylinder_ptr->mat))
            {
                gpu_c.material_data.type = 0; // Lambertian
                gpu_c.material_data.lambertian_data.albedo = lam_mat->albedo;
            }
            else if (auto metal_mat = std::dynamic_pointer_cast<metal>(cylinder_ptr->mat))
            {
                gpu_c.material_data.type = 1; // Metal
                gpu_c.material_data.metal_data.albedo = metal_mat->albedo;
                gpu_c.material_data.metal_data.fuzz = metal_mat->fuzz;
            }
            else if (auto die_mat = std::dynamic_pointer_cast<dielectric>(cylinder_ptr->mat))
            {
                gpu_c.material_data.type = 2; // Dielectric
                gpu_c.material_data.dielectric_data.ir = die_mat->ir;
            }
            else
            {
                gpu_c.material_data.type = 0;
                gpu_c.material_data.lambertian_data.albedo = color(0, 0, 0);
            }
            host_cylinders.push_back(gpu_c);
        }
        // Add more primitive types as needed
    }

    // Allocate device memory and copy data from host vectors
    num_spheres = host_spheres.size();
    if (num_spheres > 0)
    {
        cudaMalloc(&d_spheres, num_spheres * sizeof(GPUSphere));
        CudaCheckError();
        cudaMemcpy(d_spheres, host_spheres.data(), num_spheres * sizeof(GPUSphere), cudaMemcpyHostToDevice);
        CudaCheckError();
    }

    num_planes = host_planes.size();
    if (num_planes > 0)
    {
        cudaMalloc(&d_planes, num_planes * sizeof(GPUPlane));
        CudaCheckError();
        cudaMemcpy(d_planes, host_planes.data(), num_planes * sizeof(GPUPlane), cudaMemcpyHostToDevice);
        CudaCheckError();
    }

    num_boxes = host_boxes.size();
    if (num_boxes > 0)
    {
        cudaMalloc(&d_boxes, num_boxes * sizeof(GPUBox));
        CudaCheckError();
        cudaMemcpy(d_boxes, host_boxes.data(), num_boxes * sizeof(GPUBox), cudaMemcpyHostToDevice);
        CudaCheckError();
    }

    num_cylinders = host_cylinders.size();
    if (num_cylinders > 0)
    {
        cudaMalloc(&d_cylinders, num_cylinders * sizeof(GPUCylinder));
        CudaCheckError();
        cudaMemcpy(d_cylinders, host_cylinders.data(), num_cylinders * sizeof(GPUCylinder), cudaMemcpyHostToDevice);
        CudaCheckError();
    }
}

// Helper to free device memory for scene objects
void SDLRenderer::free_gpu_scene()
{
    if (d_spheres)
    {
        cudaFree(d_spheres);
        CudaCheckError();
        d_spheres = nullptr;
    }
    if (d_planes)
    {
        cudaFree(d_planes);
        CudaCheckError();
        d_planes = nullptr;
    }
    if (d_boxes)
    {
        cudaFree(d_boxes);
        CudaCheckError();
        d_boxes = nullptr;
    }
    if (d_cylinders)
    {
        cudaFree(d_cylinders);
        CudaCheckError();
        d_cylinders = nullptr;
    }
}

// Main render method: orchestrates GPU rendering
void SDLRenderer::render(const camera &cam, const hittable_list &world)
{
    // 1. Setup scene data on GPU (convert polymorphic objects to flat GPU structs)
    setup_gpu_scene(world);

    // 2. Prepare GPUCamera parameters from the host camera object
    GPUCamera gpu_cam_params;
    // The camera is already initialized in main.cu before render is called. This call is redundant and violates const.

    // Copy relevant camera parameters to the GPUCamera struct
    gpu_cam_params.center = cam.center;
    gpu_cam_params.pixel00_loc = cam.pixel00_loc;
    gpu_cam_params.pixel_delta_u = cam.pixel_delta_u;
    gpu_cam_params.pixel_delta_v = cam.pixel_delta_v;
    // Assuming 'u', 'v', 'w', 'defocus_disk_u', 'defocus_disk_v', 'time0', 'time1'
    // are now accessible public members of the camera class or can be derived.
    // If not, you'll need to adapt this based on your camera's public interface.
    gpu_cam_params.u = cam.u;
    gpu_cam_params.v = cam.v;
    gpu_cam_params.w = cam.w;
    gpu_cam_params.defocus_disk_u = cam.defocus_disk_u;
    gpu_cam_params.defocus_disk_v = cam.defocus_disk_v;
    gpu_cam_params.time0 = cam.time0;
    gpu_cam_params.time1 = cam.time1;

    // 3. Configure CUDA grid and block dimensions for the kernel launch
    // Each thread calculates one pixel. We use 16x16 threads per block.
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    dim3 threads(16, 16);

    // 4. Launch the CUDA kernel to perform raytracing on the GPU
    render_kernel<<<blocks, threads>>>(
        d_pixels,                   // Output pixel buffer on device
        width, height,              // Image dimensions
        gpu_cam_params,             // Camera parameters for the GPU
        d_spheres, num_spheres,     // Sphere data and count
        d_planes, num_planes,       // Plane data and count
        d_boxes, num_boxes,         // Box data and count
        d_cylinders, num_cylinders, // Cylinder data and count
        d_rand_states,              // CURAND states for randomness on device
        cam.samples_per_pixel,      // Samples per pixel for anti-aliasing
        cam.max_depth               // Max ray bounce depth
    );
    CudaCheckError(); // Check for errors during kernel launch

    // 5. Synchronize with the GPU and copy rendered pixels back to host memory
    cudaDeviceSynchronize(); // Wait for the kernel to complete
    CudaCheckError();
    cudaMemcpy(pixels_gpu_result.data(), d_pixels, width * height * sizeof(color), cudaMemcpyDeviceToHost);
    CudaCheckError();

    // 6. Free scene-specific GPU memory after rendering is complete
    // This is done per frame. For static scenes, you might allocate once.
    free_gpu_scene();

    // 7. Convert float colors (from GPU) to uint32_t (for SDL texture) on the host side
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            color pixel_color_f = pixels_gpu_result[y * width + x]; // Read from the correct buffer
            int ir = static_cast<int>(255.999 * pixel_color_f.x());
            int ig = static_cast<int>(255.999 * pixel_color_f.y());
            int ib = static_cast<int>(255.999 * pixel_color_f.z());
            pixels_host[y * width + x] = (255 << 24) | (ir << 16) | (ig << 8) | ib; // Write to the final texture buffer
        }
    }
}

// Present method: Updates the SDL texture and renders to the window
void SDLRenderer::present()
{
    SDL_UpdateTexture(texture, nullptr, pixels_host.data(), width * sizeof(uint32_t));
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr); // SDL automatically scales the texture to the window size
    SDL_RenderPresent(renderer);
}

// Event processing: Handles SDL window events
bool SDLRenderer::process_events()
{
    SDL_Event e;
    while (SDL_PollEvent(&e))
    {
        if (e.type == SDL_QUIT)
            return false;
        // Allows quitting with 'Q' key
        if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_q)
            return false;
    }
    return true;
}

// Poll key: Checks for key presses with a timeout
std::optional<SDL_Keycode> SDLRenderer::poll_key()
{
    SDL_Event e;
    // Wait for up to 33ms for an event (roughly 30 FPS for event polling)
    if (SDL_WaitEventTimeout(&e, 33))
    {
        if (e.type == SDL_KEYDOWN)
        {
            return e.key.keysym.sym;
        }
    }
    return std::nullopt;
}