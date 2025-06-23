#include "rtweekend.h"

#include "camera.h"
#include "hittable_list.h" // Now using hittable_list
#include "material.h"
#include "sphere.h"
#include "sdl_renderer.h"
#include "plane.h"
#include "box.h"
#include "cylinder.h"
#include "movable.h"

int main()
{
    // Changed from hittable_map to hittable_list for compatibility with SDLRenderer
    hittable_map world;

    // Materiais
    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_wall = make_shared<lambertian>(color(0.7, 0.4, 0.2));
    auto material_roof = make_shared<lambertian>(color(0.5, 0.1, 0.1));
    auto material_window = make_shared<metal>(color(0.7, 0.8, 0.9), 0.0); // Added fuzz to metal
    auto material_door = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    auto material_chimney = make_shared<lambertian>(color(0.3, 0.3, 0.3));
    auto material_glass = make_shared<metal>(1.5); // Example dielectric material

    // Chão
    world.add(make_shared<plane>(point3(0, -0.5, 0), vec3(0, 1, 0), material_ground));

    // Corpo da casa (caixa)
    world.add(make_shared<box>(point3(-0.5, -0.5, -2), point3(0.5, 0.3, -1.2), material_wall));

    // Porta (caixa pequena)
    world.add(make_shared<box>(point3(-0.08, -0.5, -1.21), point3(0.08, 0.0, -1.19), material_door));

    // Janela (caixa pequena, material metal)
    world.add(make_shared<box>(point3(0.22, -0.1, -1.21), point3(0.38, 0.1, -1.19), material_window));

    // Chaminé (cilindro)
    world.add(make_shared<cylinder>(point3(0.35, 0.3, -1.5), 0.05, 0.25, material_chimney));

    // Example sphere with glass material
    world.add(make_shared<sphere>(point3(-0.25, 0.1, -1.0), 0.1, material_glass));

    auto player = make_shared<sphere>(point3(0, 0, -1), 0.05, make_shared<lambertian>(color(0.8, 0.2, 0.2)));
    world.add(player);

    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 800;       // Increased resolution for better output
    cam.samples_per_pixel = 100; // Increased samples for smoother output
    cam.max_depth = 50;
    cam.center = point3(0, 0, 0.5);
    cam.lookat = point3(0, 0, -1.5);

    // Camera defocus and motion blur settings (example values)
    cam.defocus_angle = 0.6; // Small angle for depth of field
    cam.focus_dist = 1.8;    // Focus near the house
    cam.time0 = 0.0;
    cam.time1 = 1.0;

    // IMPORTANT: Initialize camera parameters once before the rendering loop
    cam.initialize();

    int image_height = int(cam.image_width / cam.aspect_ratio);
    int window_width = 1280; // SDL window resolution
    int window_height = int(window_width / cam.aspect_ratio);
    SDLRenderer renderer(cam.image_width, image_height, window_width, window_height);

    char moving = 's'; // 's' for scene/player movement, 'c' for camera movement
    while (renderer.process_events())
    {
        renderer.render(cam, world); // Render scene using CUDA-accelerated renderer
        renderer.present();          // Present rendered image to SDL window

        auto move = renderer.poll_key(); // Poll for key events without blocking indefinitely
        if (move)
        {
            if (*move == 'x') // Exit on 'x'
            {
                exit(0);
            }
            else if (*move == 'c') // Switch to camera movement mode
            {
                moving = 'c';
                std::cout << "Camera movement mode\n";
            }
            else if (*move == 'p') // Switch to player movement mode
            {
                moving = 's';
                std::cout << "Player movement mode\n";
            }
            else if (moving == 's') // Move player if in player mode
            {
                player->move(*move);
            }
            else if (moving == 'c') // Move camera if in camera mode
            {
                cam.move(*move);
            }
            // Re-initialize camera after movement to update internal vectors
            cam.initialize();
        }
    }
    return 0;
}