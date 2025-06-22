#include "rtweekend.h"

#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

#include <iostream>
#include <termios.h>
#include <unistd.h>

char get_char() {
    char val;
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    read(STDIN_FILENO, &val, 1);
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    std::cout << val << std::endl;
    return val;
}

int main () {
    hittable_list world;

    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto material_left   = make_shared<metal>(color(0.8, 0.8, 0.8));
    auto material_right  = make_shared<metal>(color(0.8, 0.6, 0.2));

    auto movable_sphere = make_shared<sphere>(point3( 0.0,    0.0, -1.2),   0.5, material_center);

    world.add(make_shared<sphere>(point3( 0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(movable_sphere);
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.5, material_left));
    world.add(make_shared<sphere>(point3( 1.0,    0.0, -1.0),   0.5, material_right));

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 80;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 10;

    while (true) {
        char move = get_char();
        if (move == 'q') {
            exit(0);
        } else {
            movable_sphere->move(move);
        }

        cam.render(world, "image.ppm");
    }
}
