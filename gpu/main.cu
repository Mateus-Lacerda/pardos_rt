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
#include <chrono>
#include <iomanip>
#include <fstream>
#include <limits>

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

    // Variáveis para o benchmark
    long long total_frames = 0;
    double total_duration_ms = 0.0;
    double min_frame_time = std::numeric_limits<double>::max();
    double max_frame_time = 0.0;
    while (renderer.process_events())
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        renderer.render(cam, world);
        renderer.present();

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> frame_duration = end_time - start_time;
        double current_frame_time_ms = frame_duration.count();

        // Atualiza as estatísticas do benchmark
        total_frames++;
        total_duration_ms += current_frame_time_ms;
        min_frame_time = std::min(min_frame_time, current_frame_time_ms);
        max_frame_time = std::max(max_frame_time, current_frame_time_ms);
        auto move_keycode = renderer.poll_key();
        if (move_keycode)
        {
            char move_char = static_cast<char>(*move_keycode);

            // Limpa a linha do console antes de imprimir o novo status
            std::cout << "\r" << std::string(50, ' ') << "\r";

            switch (move_char)
            {
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
            if (move_char == 'x')
            {
                std::cout << "Exiting benchmark..." << std::endl;
                break;
            }
        }

        std::cout << "\rFrame time: " << std::fixed << std::setprecision(2) << current_frame_time_ms << " ms | Mode: " << (moving == 'c' ? "Camera" : "Player") << std::flush;
    }

    // Imprime uma nova linha no final para limpar a saída do console
    std::cout << std::endl;

    // Salva os resultados do benchmark em um arquivo
    if (total_frames > 0)
    {
        double average_frame_time = total_duration_ms / total_frames;
        double average_fps = (average_frame_time > 0) ? 1000.0 / average_frame_time : 0;
        double min_fps = (max_frame_time > 0) ? 1000.0 / max_frame_time : 0;
        double max_fps = (min_frame_time > 0) ? 1000.0 / min_frame_time : 0;

        std::ofstream results_file("benchmark_results.txt");
        if (results_file.is_open())
        {
            results_file << "--- CUDA Raytracer Benchmark Results ---\n";
            results_file << "Image Dimensions: " << cam.image_width << "x" << image_height << "\n";
            results_file << "Samples Per Pixel: " << cam.samples_per_pixel << "\n";
            results_file << "Max Ray Depth: " << cam.max_depth << "\n";
            results_file << "----------------------------------------\n";
            results_file << "Total Frames Rendered: " << total_frames << "\n";
            results_file << "Average Frame Time (ms): " << std::fixed << std::setprecision(2) << average_frame_time << "\n";
            results_file << "Minimum Frame Time (ms): " << std::fixed << std::setprecision(2) << min_frame_time << "\n";
            results_file << "Maximum Frame Time (ms): " << std::fixed << std::setprecision(2) << max_frame_time << "\n";
            results_file << "----------------------------------------\n";
            results_file << "Average FPS: " << std::fixed << std::setprecision(2) << average_fps << "\n";
            results_file << "Minimum FPS: " << std::fixed << std::setprecision(2) << min_fps << "\n";
            results_file << "Maximum FPS: " << std::fixed << std::setprecision(2) << max_fps << "\n";
            results_file << "----------------------------------------\n";
            results_file.close();
            std::cout << "Benchmark results saved to benchmark_results.txt" << std::endl;
        }
    }
    return 0;
}