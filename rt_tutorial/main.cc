#include "rtweekend.h"

#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "sdl_renderer.h"
#include "plane.h"
#include "box.h"
#include "cylinder.h"
#include "movable.h"

int main()
{
    hittable_map world;

    // Materiais
    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_wall = make_shared<lambertian>(color(0.7, 0.4, 0.2));
    auto material_roof = make_shared<lambertian>(color(0.5, 0.1, 0.1));
    auto material_window = make_shared<metal>(color(0.7, 0.8, 0.9));
    auto material_door = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    auto material_chimney = make_shared<lambertian>(color(0.3, 0.3, 0.3));

    // Chão
    world.add(make_shared<plane>(point3(0, -0.5, 0), vec3(0, 1, 0), material_ground));

    // Corpo da casa (caixa)
    world.add(make_shared<box>(point3(-0.5, -0.5, -2), point3(0.5, 0.3, -1.2), material_wall));

    // // Telhado (triângulo com 2 planos inclinados)
    // world.add(make_shared<plane>(point3(-0.5, 0.3, -2), vec3(0.5, 0.7, 0.4), material_roof));
    // world.add(make_shared<plane>(point3(0.5, 0.3, -2), vec3(-0.5, 0.7, 0.4), material_roof));

    // Porta (caixa pequena)
    world.add(make_shared<box>(point3(-0.08, -0.5, -1.21), point3(0.08, 0.0, -1.19), material_door));

    // Janela (caixa pequena, material metal)
    world.add(make_shared<box>(point3(0.22, -0.1, -1.21), point3(0.38, 0.1, -1.19), material_window));

    // Chaminé (cilindro)
    world.add(make_shared<cylinder>(point3(0.35, 0.3, -1.5), 0.05, 0.25, material_chimney));

    // Esfera decorativa no topo
    // world.add(make_shared<sphere>(point3(0, 0.55, -1.6), 0.07, material_roof));

    auto player = make_shared<sphere>(point3(0, 0, -1), 0.05, make_shared<lambertian>(color(0.8, 0.2, 0.2)));
    world.add(player);

    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 80;
    cam.samples_per_pixel = 40;
    cam.max_depth = 10;
    cam.center = point3(0, 0, 0.5);
    cam.lookat = point3(0, 0, -1.5);

    int image_height = int(cam.image_width / cam.aspect_ratio);
    int window_width = 800;
    int window_height = int(window_width / cam.aspect_ratio);
    SDLRenderer renderer(cam.image_width, image_height, window_width, window_height);

    char moving = 's';
    while (renderer.process_events())
    {
        renderer.render(cam, world);
        renderer.present();
        auto move = get_char(0.16);
        if (move) {
            if (*move == 'x')
            {
                exit(0);
            }
            else if (*move == 'c')
            {
                moving = 'c';
            }
            else if (*move == 'p')
            {
                moving = 's';
            }
            else if (moving == 's')
            {
                player->move(*move);
            }
            else if (moving == 'c')
            {
                cam.move(*move);
            }
        }
    }
    return 0;
}
