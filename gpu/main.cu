// Ordem de includes corrigida para garantir que as definições completas
// das classes sejam visíveis antes de serem usadas.
#include "rtweekend.h"

#include "camera.h"
#include "hittable_list.h" // Definição completa primeiro
#include "material.h"      // Definição completa primeiro
#include "sdl_renderer.h"  // Agora o renderer, que usa as declarações
#include "sphere.h"
#include "plane.h"
#include "box.h"
#include "cylinder.h"
#include "movable.h"

#include <iostream>

int main()
{
    hittable_list world;

    // Materiais
    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_wall = make_shared<lambertian>(color(0.7, 0.4, 0.2));
    auto material_roof = make_shared<lambertian>(color(0.5, 0.1, 0.1));
    auto material_window = make_shared<metal>(color(0.7, 0.8, 0.9), 0.0);
    auto material_door = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    auto material_chimney = make_shared<lambertian>(color(0.3, 0.3, 0.3));
    auto material_glass = make_shared<dielectric>(1.5);

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

    // Esfera de vidro
    world.add(make_shared<sphere>(point3(-0.25, 0.1, -1.0), 0.1, material_glass));

    // Personagem/Jogador
    auto player = make_shared<sphere>(point3(0, 0, -1), 0.05, make_shared<lambertian>(color(0.8, 0.2, 0.2)));
    world.add(player);

    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 80;
    cam.samples_per_pixel = 40;
    cam.max_depth = 10;
    cam.center = point3(0, 0, 0.5);
    cam.lookat = point3(0, 0, -1.5);
    cam.defocus_angle = 0.6;
    cam.focus_dist = 1.8;
    cam.time0 = 0.0;
    cam.time1 = 1.0;

    cam.initialize();

    int image_height = static_cast<int>(cam.image_width / cam.aspect_ratio);
    int window_width = 1280;
    int window_height = static_cast<int>(window_width / cam.aspect_ratio);
    SDLRenderer renderer(cam.image_width, image_height, window_width, window_height);

    char moving = 's'; // 's' para sphere (player), 'c' para camera
    while (renderer.process_events())
    {
        renderer.render(cam, world);
        renderer.present();

        auto move_keycode = renderer.poll_key();
        if (move_keycode)
        {
            char move_char = static_cast<char>(*move_keycode);

            switch (move_char)
            {
            case 'x':
                exit(0);
                break;
            case 'c':
                moving = 'c';
                std::cout << "Camera movement mode\n";
                break;
            case 'p':
                moving = 's';
                std::cout << "Player movement mode\n";
                break;
            default:
                if (moving == 's')
                {
                    player->move(move_char);
                }
                else if (moving == 'c')
                {
                    cam.move(move_char);
                }
                cam.initialize(); // Re-inicializa a câmera após qualquer movimento
                break;
            }
        }
    }

    return 0;
}