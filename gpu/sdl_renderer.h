#ifndef SDL_RENDERER_H
#define SDL_RENDERER_H

#include "hittable_list.h" // For hittable_list
#include "camera.h"        // For Camera class parameters
#include "hittable.h"      // For GPUSphere, GPUPlane, GPUBox, GPUCylinder, GPUMaterial, color
#include "material.h"      // For GPULambertian, GPUMetal, GPUDielectric, and GPUMaterial
#include <SDL2/SDL_keycode.h>
#include <vector>
#include <cstdint>
#include <optional>

// **CRITICAL FIX: Wrap CUDA includes with #ifdef __CUDACC__**
// These headers are specific to NVCC compilation (device code).
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <curand.h> // For curand host-side functions (e.g., curandCreateGenerator)
// curand_kernel.h is only needed in the .cu file for __device__ functions.
// It should not be included here.
#endif

// Forward declaration of GPUCamera struct (defined in cuda_raytracer_kernel.cu)
// This declaration does NOT need __device__ as it's a data structure.
struct GPUCamera;

class SDLRenderer
{
public:
    SDLRenderer(int img_width, int img_height, int win_width, int win_height);
    ~SDLRenderer();

    void render(const camera &cam, const hittable_list &world);
    void present();
    bool process_events();
    std::optional<SDL_Keycode> poll_key();

    int width, height;               // image size (resolution of the raytraced image)
    int window_width, window_height; // window size (resolution of the SDL window)

private:
    struct SDL_Window *window;
    struct SDL_Renderer *renderer;
    struct SDL_Texture *texture;
    std::vector<uint32_t> pixels_host; // Host-side pixel buffer for SDL texture update

    // CUDA Device Pointers (only allocated if __CUDACC__ is defined, but pointers are always members)
    // To avoid issues with g++ knowing cudaError_t, etc., we can declare these as void*
    // or rely on the include guards, which is what we're doing.
    color *d_pixels;            // Device-side pixel buffer (RGBA for colors)
    curandState *d_rand_states; // Device-side CURAND states for each thread

    // Device-side scene data pointers
    GPUSphere *d_spheres;
    GPUPlane *d_planes;
    GPUBox *d_boxes;
    GPUCylinder *d_cylinders;

    // Counts of scene objects
    int num_spheres;
    int num_planes;
    int num_boxes;
    int num_cylinders;

    // Helper to setup GPU scene data from host hittable_list
    void setup_gpu_scene(const hittable_list &world);
    // Helper to free GPU scene data
    void free_gpu_scene();
    // Helper to initialize CURAND states on device (no longer needed as separate method if in kernel)
    void init_curand(); // Keep declaration, but implementation will be empty or removed if not needed.
};

#endif // !SDL_RENDERER_H