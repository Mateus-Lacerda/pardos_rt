#ifndef SDL_RENDERER_H
#define SDL_RENDERER_H

#include <SDL2/SDL_keycode.h>
#include <vector>
#include <cstdint>
#include <optional>
#include "color.h"

// Forward declarations
class camera;
class hittable_list;
struct GPUSphere;
struct GPUPlane;
struct GPUBox;
struct GPUCylinder;

// Forward declare the curandState struct for host-side code.

class SDLRenderer
{
public:
    SDLRenderer(int img_width, int img_height, int win_width, int win_height);
    ~SDLRenderer();

    void render(const camera &cam, const hittable_list &world);
    void present();
    bool process_events();
    std::optional<SDL_Keycode> poll_key();

    int width, height;
    int window_width, window_height;

private:
    struct SDL_Window *window;
    struct SDL_Renderer *renderer;
    struct SDL_Texture *texture;
    std::vector<uint32_t> pixels_host;
    std::vector<color> pixels_gpu_result; // New buffer for raw GPU results

    color *d_pixels;
    // <<< FIX: Removido o 'struct' daqui >>>
    curandState *d_rand_states;
    GPUSphere *d_spheres;
    GPUPlane *d_planes;
    GPUBox *d_boxes;
    GPUCylinder *d_cylinders;

    int num_spheres;
    int num_planes;
    int num_boxes;
    int num_cylinders;

    void setup_gpu_scene(const hittable_list &world);
    void free_gpu_scene();
    void init_curand();
};

#endif