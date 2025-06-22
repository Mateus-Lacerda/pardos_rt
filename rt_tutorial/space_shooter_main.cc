#include "space_shooter_game.h"
#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "sdl_renderer.h"
#include "plane.h"
#include "box.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main()
{
    while (true)
    {
        std::srand(std::time(nullptr));
        hittable_list world;
        auto material_ground = make_shared<lambertian>(color(0.05, 0.05, 0.1));
        world.add(make_shared<plane>(point3(0, -0.5, 0), vec3(0, 1, 0), material_ground));
        auto player = make_shared<sphere>(point3(0, -0.45, -1), 0.1, make_shared<metal>(color(0.2, 0.8, 1.0)));
        world.add(player);

        camera cam;
        cam.aspect_ratio = 16.0 / 9.0;
        cam.image_width = 120;
        cam.samples_per_pixel = 20;
        cam.max_depth = 20;
        cam.center = point3(0, 0, -0.5);
        cam.lookat = point3(0, -0.2, -1.5);

        int image_height = int(cam.image_width / cam.aspect_ratio);
        int window_width = 800;
        int window_height = int(window_width / cam.aspect_ratio);
        SDLRenderer renderer(cam.image_width, image_height, window_width, window_height);

        SpaceShooterGame game(&world, player);
        int frame = 0;
        while (renderer.process_events())
        {
            renderer.render(cam, world);
            renderer.present();
            if (game.game_over)
            {
                std::cout << "Game Over! Score: " << game.score << ". Press x to exit or r to restart.\n";
                auto move = renderer.poll_key();
                if (move)
                {
                    if (*move == SDLK_x)
                        return 0;
                    else if (*move == SDLK_r)
                        break;
                }
                continue;
            }
            auto move = renderer.poll_key();
            if (move)
            {
                if (*move == SDLK_x)
                    return 0;
            }
            game.update(move, frame);
            frame++;
        }
    }
    return 0;
}
