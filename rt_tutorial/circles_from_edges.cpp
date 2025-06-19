// circles_from_edges.cpp
// Gera círculos máximos a partir de uma imagem PPM P3 de cantos (edges)
// e exporta no formato do seu ray tracing
//
// Compile com: g++ -o circles_from_edges circles_from_edges.cpp
// Uso: ./circles_from_edges edges.ppm original.ppm output_spheres.h

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>

struct Pixel
{
    int r, g, b;
};

struct Circle
{
    double x, y, r;
    double cr, cg, cb;
};

// Parâmetros do viewport do ray tracer (ajuste conforme sua câmera)
double viewport_height = 2.0;
double viewport_width;

// Função para ler PPM P3
std::vector<std::vector<Pixel>> read_ppm(const std::string &filename, int &width, int &height)
{
    std::ifstream f(filename);
    std::string magic;
    f >> magic;
    while (f.peek() == '\n' || f.peek() == '#')
        f.ignore(2048, '\n');
    f >> width >> height;
    int maxval;
    f >> maxval;
    std::vector<std::vector<Pixel>> img(height, std::vector<Pixel>(width));
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            int r, g, b;
            f >> r >> g >> b;
            img[i][j] = {r, g, b};
        }
    }
    return img;
}

// Função para binarizar a imagem de cantos
std::vector<std::vector<bool>> threshold_edges(const std::vector<std::vector<Pixel>> &edges, int threshold = 30)
{
    int h = edges.size(), w = edges[0].size();
    std::vector<std::vector<bool>> mask(h, std::vector<bool>(w, false));
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            int v = (edges[i][j].r + edges[i][j].g + edges[i][j].b) / 3;
            mask[i][j] = (v > threshold); // true = canto
        }
    }
    return mask;
}

// Checa se um círculo cabe sem tocar cantos
bool can_place(const std::vector<std::vector<bool>> &mask, int x, int y, int r)
{
    int h = mask.size(), w = mask[0].size();
    for (int dy = -r; dy <= r; ++dy)
    {
        for (int dx = -r; dx <= r; ++dx)
        {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                return false;
            if (dx * dx + dy * dy > r * r)
                continue;
            if (mask[ny][nx])
                return false;
        }
    }
    return true;
}

// Marca pixels cobertos por um círculo
void mark_circle(std::vector<std::vector<bool>> &mask, int x, int y, int r)
{
    int h = mask.size(), w = mask[0].size();
    for (int dy = -r; dy <= r; ++dy)
    {
        for (int dx = -r; dx <= r; ++dx)
        {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                continue;
            if (dx * dx + dy * dy > r * r)
                continue;
            mask[ny][nx] = true;
        }
    }
}

// Calcula cor média do círculo
std::tuple<double, double, double> avg_color(const std::vector<std::vector<Pixel>> &img, int x, int y, int r)
{
    int h = img.size(), w = img[0].size();
    double sr = 0, sg = 0, sb = 0;
    int n = 0;
    for (int dy = -r; dy <= r; ++dy)
    {
        for (int dx = -r; dx <= r; ++dx)
        {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                continue;
            if (dx * dx + dy * dy > r * r)
                continue;
            sr += img[ny][nx].r;
            sg += img[ny][nx].g;
            sb += img[ny][nx].b;
            n++;
        }
    }
    if (n == 0)
        return {1, 1, 1};
    return {sr / n / 255.0, sg / n / 255.0, sb / n / 255.0};
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Uso: " << argv[0] << " <edges.ppm> <original.ppm> <output_spheres.h>\n";
        return 1;
    }
    int w, h;
    auto edges = read_ppm(argv[1], w, h);
    auto mask = threshold_edges(edges);
    int w2, h2;
    auto img = read_ppm(argv[2], w2, h2);
    if (w != w2 || h != h2)
    {
        std::cerr << "Dimensões diferentes entre edges e original!\n";
        return 1;
    }
    viewport_width = viewport_height * (double(w) / h); // Ex: 16:9
    std::vector<Circle> circles;
    std::vector<std::vector<bool>> covered = mask;
    int min_r = 6, max_r = std::min(w, h) / 6;
    int step = 8; // Ajuste para mais (menos círculos) ou menos (mais detalhes)
    // Função para mapear coordenadas de pixel para viewport
    auto pixel_to_viewport = [&](int x, int y)
    {
        double x_vp = (double(x) / w) * viewport_width - viewport_width / 2.0;
        double y_vp = -((double(y) / h) * viewport_height - viewport_height / 2.0);
        return std::make_pair(x_vp, y_vp);
    };
    for (int y = 0; y < h; y += step)
    {
        for (int x = 0; x < w; x += step)
        {
            if (covered[y][x])
                continue;
            int r = max_r;
            for (; r >= min_r; --r)
            {
                if (can_place(covered, x, y, r))
                {
                    auto [cr, cg, cb] = avg_color(img, x, y, r);
                    auto [x_vp, y_vp] = pixel_to_viewport(x, y);
                    // O raio também pode ser normalizado se quiser
                    circles.push_back({x_vp, y_vp, r * viewport_width / w, cr, cg, cb});
                    mark_circle(covered, x, y, r);
                    break;
                }
            }
        }
    }
    std::ofstream out(argv[3]);
    out << "// Esferas geradas a partir dos círculos\n";
    for (const auto &c : circles)
    {
        out << "world.add(make_shared<sphere>(point3(" << c.x << ", " << c.y << ", -1), " << c.r << ", make_shared<lambertian>(color(" << c.cr << ", " << c.cg << ", " << c.cb << "))));\n";
    }
    std::cout << "Gerado " << circles.size() << " esferas em " << argv[3] << "\n";
    return 0;
}
