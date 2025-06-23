#ifndef SPACE_SHOOTER_GAME_H
#define SPACE_SHOOTER_GAME_H

#include "hittable_list.h"
#include "sphere.h"
#include "box.h"
#include <SDL2/SDL_keycode.h>
#include <vector>
#include <memory>
#include <cstdlib>

class SpaceShooterGame
{
public:
    shared_ptr<sphere> player;
    std::vector<shared_ptr<box>> enemies;
    std::vector<shared_ptr<sphere>> shots;
    hittable_map *world;
    double enemy_speed = 0.025;
    double shot_speed = 0.07;
    double spawn_interval = 30; // frames
    bool game_over = false;
    int score = 0;

    SpaceShooterGame(hittable_map *world, shared_ptr<sphere> player)
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
            if (*input == SDLK_w)
            {
                // Dispara tiro
                auto shot = make_shared<sphere>(point3(player->center[0], player->center[1] + 0.09, player->center[2]), 0.02, make_shared<lambertian>(color(1, 1, 0.2)));
                shots.push_back(shot);
                world->add(shot);
            }
        }
        // Limita área do player
        if (player->center[0] < -0.45)
            player->center[0] = -0.45;
        if (player->center[0] > 0.45)
            player->center[0] = 0.45;
        // Move inimigos
        for (auto &e : enemies)
        {
            e->min_corner[1] -= enemy_speed;
            e->max_corner[1] -= enemy_speed;
        }
        // Move tiros
        for (auto &s : shots)
        {
            s->center[1] += shot_speed;
        }
        // Remove inimigos e tiros fora da tela
        enemies.erase(std::remove_if(enemies.begin(), enemies.end(), [&](auto &e)
                                     { return e->max_corner[1] < -0.6; }),
                      enemies.end());
        shots.erase(std::remove_if(shots.begin(), shots.end(), [&](auto &s)
                                   { return s->center[1] > 0.7; }),
                    shots.end());
        // Spawn de novos inimigos
        if (frame % (int)spawn_interval == 0)
        {
            double x = -0.4 + 0.8 * (rand() / (double)RAND_MAX);
            auto mat = make_shared<lambertian>(color(0.8, 0.2, 0.2));
            auto enemy = make_shared<box>(point3(x, 0.6, -1), point3(x + 0.12, 0.68, -0.9), mat);
            enemies.push_back(enemy);
            world->add(enemy);
        }
        // Colisão tiro-inimigo (marcar para remoção)
        std::vector<shared_ptr<box>> enemies_to_remove;
        std::vector<shared_ptr<sphere>> shots_to_remove;
        for (auto &s : shots)
        {
            for (auto &e : enemies)
            {
                if (collides(s, e))
                {
                    enemies_to_remove.push_back(e);
                    shots_to_remove.push_back(s);
                    score++;
                }
            }
        }
        // Remove inimigos e tiros atingidos
        enemies.erase(std::remove_if(enemies.begin(), enemies.end(), [&](auto &e)
                                     { return std::find(enemies_to_remove.begin(), enemies_to_remove.end(), e) != enemies_to_remove.end(); }),
                      enemies.end());
        shots.erase(std::remove_if(shots.begin(), shots.end(), [&](auto &s)
                                   { return std::find(shots_to_remove.begin(), shots_to_remove.end(), s) != shots_to_remove.end(); }),
                    shots.end());
        for (auto &e : enemies_to_remove) {
            world->remove(e->id()); // Certifique-se de que 'id' existe em box
        }
        for (auto &s : shots_to_remove) {
            world->remove(s->id()); // Certifique-se de que 'id' existe em box
        }
        // Colisão player-inimigo
        for (auto &e : enemies)
        {
            if (collides(player, e))
            {
                game_over = true;
            }
        }
    }

    bool collides(shared_ptr<sphere> s, shared_ptr<box> b)
    {
        double x = std::max(b->min_corner[0], std::min(s->center[0], b->max_corner[0]));
        double y = std::max(b->min_corner[1], std::min(s->center[1], b->max_corner[1]));
        double z = std::max(b->min_corner[2], std::min(s->center[2], b->max_corner[2]));
        double dist2 = (x - s->center[0]) * (x - s->center[0]) + (y - s->center[1]) * (y - s->center[1]) + (z - s->center[2]) * (z - s->center[2]);
        return dist2 < s->get_radius() * s->get_radius();
    }
};

#endif // SPACE_SHOOTER_GAME_H
