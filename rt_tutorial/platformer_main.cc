#include "platformer_game.h"
#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "sdl_renderer.h"
#include "plane.h"
#include "box.h"
#include "cylinder.h"
#include <iostream>

int main()
{
    // Cria o mundo e player igual ao main.cc
    hittable_map world;
    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_wall = make_shared<lambertian>(color(0.7, 0.4, 0.2));
    auto material_roof = make_shared<lambertian>(color(0.5, 0.1, 0.1));
    auto material_window = make_shared<metal>(color(0.7, 0.8, 0.9));
    auto material_door = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    auto material_chimney = make_shared<lambertian>(color(0.3, 0.3, 0.3));
    auto material_lake = make_shared<metal>(color(0.2, 0.5, 0.8));
    world.add(make_shared<plane>(point3(0, -0.5, 0), vec3(0, 1, 0), material_ground));
    world.add(make_shared<box>(point3(-0.5, -0.5, -2), point3(0.5, 0.3, -1.2), material_wall));
    world.add(make_shared<box>(point3(-0.08, -0.5, -1.21), point3(0.08, 0.0, -1.19), material_door));
    world.add(make_shared<box>(point3(0.22, -0.1, -1.21), point3(0.38, 0.1, -1.19), material_window));
    world.add(make_shared<cylinder>(point3(0.35, 0.3, -1.5), 0.05, 0.25, material_chimney));
    world.add(make_shared<plane>(point3(0.3, -0.49, -0.7), vec3(0, 1, 0), material_lake));
    auto player = make_shared<sphere>(point3(0, 0, -0.2), 0.05, make_shared<lambertian>(color(0.8, 0.2, 0.2)));
    world.add(player);

    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 120;
    cam.samples_per_pixel = 40;
    cam.max_depth = 10;
    cam.center = point3(0, 0, 0.5);
    cam.lookat = point3(0, 0, -1.5);

    int image_height = int(cam.image_width / cam.aspect_ratio);
    int window_width = 800;
    int window_height = int(window_width / cam.aspect_ratio);
    SDLRenderer renderer(cam.image_width, image_height, window_width, window_height);

    PlatformerGame game(&world, player);

    while (renderer.process_events())
    {
        renderer.render(cam, world);
        renderer.present();
        auto move = renderer.poll_key();
        if (move) {
            if (*move == SDLK_x) return 0;
        }
        game.update(move);
    }
    return 0;
}
