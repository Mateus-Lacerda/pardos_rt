#ifndef DODGE_GAME_H
#define DODGE_GAME_H

#include "hittable_list.h"
#include "sphere.h"
#include "box.h"
#include <SDL2/SDL_keycode.h>
#include <vector>
#include <memory>
#include <cstdlib>

class DodgeGame
{
public:
    shared_ptr<sphere> player;
    std::vector<shared_ptr<box>> obstacles;
    hittable_list *world;
    double speed = 0.03;
    double spawn_timer = 0;
    double spawn_interval = 20; // frames
    bool game_over = false;

    DodgeGame(hittable_list *world, shared_ptr<sphere> player)
        : world(world), player(player) {}

    void update(std::optional<SDL_Keycode> input, int frame)
    {
        if (game_over)
            return;
        // Movimento do player
        if (input)
        {
            if (*input == SDLK_a)
                player->center[0] -= 0.08;
            if (*input == SDLK_d)
                player->center[0] += 0.08;
        }
        // Limita área do player
        if (player->center[0] < -0.45)
            player->center[0] = -0.45;
        if (player->center[0] > 0.45)
            player->center[0] = 0.45;
        // Move obstáculos
        for (auto &obs : obstacles)
        {
            obs->min_corner[1] -= speed;
            obs->max_corner[1] -= speed;
        }
        // Remove obstáculos que passaram
        obstacles.erase(std::remove_if(obstacles.begin(), obstacles.end(), [&](auto &obs)
                                       { return obs->max_corner[1] < -0.6; }),
                        obstacles.end());
        // Spawn de novos obstáculos
        if (frame % (int)spawn_interval == 0)
        {
            double x = -0.4 + 0.8 * (rand() / (double)RAND_MAX);
            auto mat = make_shared<lambertian>(color(0.2, 0.2, 0.8));
            auto obs = make_shared<box>(point3(x, 0.6, -1), point3(x + 0.15, 0.7, -0.9), mat);
            obstacles.push_back(obs);
            world->add(obs);
        }
        // Colisão
        for (auto &obs : obstacles)
        {
            if (collides(player, obs))
            {
                game_over = true;
            }
        }
    }

    bool collides(shared_ptr<sphere> s, shared_ptr<box> b)
    {
        // AABB vs Sphere
        double x = std::max(b->min_corner[0], std::min(s->center[0], b->max_corner[0]));
        double y = std::max(b->min_corner[1], std::min(s->center[1], b->max_corner[1]));
        double z = std::max(b->min_corner[2], std::min(s->center[2], b->max_corner[2]));
        double dist2 = (x - s->center[0]) * (x - s->center[0]) + (y - s->center[1]) * (y - s->center[1]) + (z - s->center[2]) * (z - s->center[2]);
        return dist2 < s->get_radius() * s->get_radius();
    }
};

#endif // DODGE_GAME_H
