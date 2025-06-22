#ifndef PLATFORMER_GAME_H
#define PLATFORMER_GAME_H

#include "hittable_list.h"
#include "sphere.h"
#include "box.h"
#include <memory>

class PlatformerGame
{
public:
    shared_ptr<sphere> player;
    hittable_list *world;
    double velocity_y = 0.0;
    bool on_ground = false;
    double gravity = -0.03;
    double jump_strength = 0.15;

    PlatformerGame(hittable_list *world, shared_ptr<sphere> player)
        : world(world), player(player) {}

    void update(std::optional<char> input)
    {
        // Movimento horizontal
        if (input) {
            char move = *input;
            if (move == 'a')
            {
                player->center[0] -= 0.05;
            }
            else if (move == 'd')
            {
                player->center[0] += 0.05;
            }
            // Pulo
            if (move == 'w' && on_ground)
            {
                velocity_y = jump_strength;
                on_ground = false;
            }
        }
        // Física simples
        velocity_y += gravity;
        player->center[1] += velocity_y;
        // Checa colisão com o chão (y = -0.45 para o player)
        if (player->center[1] <= -0.45)
        {
            player->center[1] = -0.45;
            velocity_y = 0;
            on_ground = true;
        }
    }
};

#endif // PLATFORMER_GAME_H
