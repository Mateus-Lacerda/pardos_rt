#include "sdl_renderer.h"
#include <SDL2/SDL.h>
#include <omp.h>
#include <vector>
#include <algorithm>

SDLRenderer::SDLRenderer(int img_w, int img_h, int win_w, int win_h)
    : width(img_w), height(img_h), window_width(win_w), window_height(win_h), pixels(img_w * img_h, 0)
{
    SDL_Init(SDL_INIT_VIDEO);
    window = SDL_CreateWindow("Raytracer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, window_width, window_height, 0);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height);
}

SDLRenderer::~SDLRenderer()
{
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void SDLRenderer::render(camera &cam, hittable_map &world)
{
    cam.initialize(); // Garante que a câmera está pronta antes do paralelismo
#pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            color pixel_color = cam.get_pixel_color(x, y, world);
            int ir = static_cast<int>(255.999 * std::clamp(pixel_color.x(), 0.0, 1.0));
            int ig = static_cast<int>(255.999 * std::clamp(pixel_color.y(), 0.0, 1.0));
            int ib = static_cast<int>(255.999 * std::clamp(pixel_color.z(), 0.0, 1.0));
            pixels[y * width + x] = (255 << 24) | (ir << 16) | (ig << 8) | ib;
        }
    }
}

void SDLRenderer::present()
{
    SDL_UpdateTexture(texture, nullptr, pixels.data(), width * sizeof(uint32_t));
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr); // SDL faz o stretch automático
    SDL_RenderPresent(renderer);
}

bool SDLRenderer::process_events()
{
    SDL_Event e;
    while (SDL_PollEvent(&e))
    {
        if (e.type == SDL_QUIT)
            return false;
        if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_q)
            return false;
    }
    return true;
}

std::optional<SDL_Keycode> SDLRenderer::poll_key()
{
    SDL_Event e;
   if (SDL_WaitEventTimeout(&e, 33)) {
        if (e.type == SDL_KEYDOWN) {
            return e.key.keysym.sym;
        }
    }
    return std::nullopt;
}
