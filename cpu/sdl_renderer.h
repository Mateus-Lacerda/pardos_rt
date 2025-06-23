#ifndef SDL_RENDERER_H
#define SDL_RENDERER_H

#include "hittable_list.h"
#include "camera.h"
#include <SDL2/SDL_keycode.h>
#include <vector>
#include <cstdint>
#include <optional>

class SDLRenderer
{
public:
    SDLRenderer(int img_width, int img_height, int win_width, int win_height);
    ~SDLRenderer();
    void render(camera &cam, hittable_map &world);
    void present();
    bool process_events();
    int width, height;               // image size
    int window_width, window_height; // window size
    std::vector<uint32_t> pixels;
    std::optional<SDL_Keycode> poll_key();

private:
    struct SDL_Window *window;
    struct SDL_Renderer *renderer;
    struct SDL_Texture *texture;
};

#endif // !SDL_RENDERER_H
