#ifndef SDL_RENDERER_H
#define SDL_RENDERER_H

#include "hittable_list.h"
#include "camera.h"
#include <vector>
#include <cstdint>

class SDLRenderer
{
public:
    SDLRenderer(int img_width, int img_height, int win_width, int win_height);
    ~SDLRenderer();
    void render(camera &cam, hittable_list &world);
    void present();
    bool process_events();
    int width, height;               // image size
    int window_width, window_height; // window size
    std::vector<uint32_t> pixels;

private:
    struct SDL_Window *window;
    struct SDL_Renderer *renderer;
    struct SDL_Texture *texture;
};

#endif // !SDL_RENDERER_H