g++ -std=c++17 -O2 -Wall -Wextra -fopenmp main.cc sdl_renderer.cc -o main $(sdl2-config --cflags --libs)
