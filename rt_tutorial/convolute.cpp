#include <iostream>
#include <stdexcept>
#include <fftw3.h>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include "matrix.h"

// Sobel vertical kernel
#define SOBEL_V Matrix({{-1, -2, -1}, \
                        {0, 0, 0},    \
                        {1, 2, 1}})

// Sobel horizontal kernel
#define SOBEL_H Matrix({{-1, 0, 1}, \
                        {-2, 0, 2}, \
                        {-1, 0, 1}})
// Função para ignorar comentários e ler cabeçalho PPM
void readPPMHeader(std::istream &input, std::string &magic, int &width, int &height, int &maxval)
{
    input >> magic;
    if (magic != "P3")
    {
        throw std::runtime_error("Formato PPM inválido (esperado P3)");
    }
    // Ignorar comentários e linhas em branco
    std::string line;
    while (true)
    {
        std::getline(input, line);
        if (line.empty() || line[0] == '#')
            continue;
        std::stringstream ss(line);
        if (ss >> width >> height)
            break;
    }
    // Ler maxval (pode estar na mesma linha ou na próxima)
    while (true)
    {
        if (!(input >> maxval))
        {
            std::getline(input, line);
            if (line.empty() || line[0] == '#')
                continue;
            std::stringstream ss(line);
            if (ss >> maxval)
                break;
        }
        else
        {
            break;
        }
    }
    if (width <= 0 || height <= 0 || maxval <= 0)
    {
        throw std::runtime_error("Cabeçalho PPM inválido");
    }
    // Consumir qualquer espaço em branco ou quebra de linha antes dos dados dos pixels
    input >> std::ws;
}

// Função utilitária para detecção de bordas/cantos usando Sobel
Matrix detectEdges(const Matrix &input)
{
    Matrix sobelV = SOBEL_V;
    Matrix sobelH = SOBEL_H;
    Matrix gradV = input.convolve2D(sobelV);
    Matrix gradH = input.convolve2D(sobelH);
    int rows = gradV.getRows();
    int cols = gradV.getCols();
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            double gx = gradH.getElement(i, j);
            double gy = gradV.getElement(i, j);
            result.setElement(i, j, std::sqrt(gx * gx + gy * gy));
        }
    }
    return result;
}

int main(int argc, char *argv[])
{
    try
    {
        // Ler imagem PPM de um arquivo ou stdin
        std::string inputFile = (argc > 1) ? argv[1] : "";
        std::ifstream input;
        if (!inputFile.empty())
        {
            input.open(inputFile);
            if (!input)
                throw std::runtime_error("Não foi possível abrir o arquivo de entrada");
        }
        std::istream &imgInput = input.is_open() ? input : std::cin;

        // Ler cabeçalho PPM
        std::string magic;
        int width, height, maxval;
        readPPMHeader(imgInput, magic, width, height, maxval);

        // Ler todos os pixels RGB em um buffer
        std::vector<int> pixels(width * height * 3);
        for (int i = 0; i < width * height * 3; ++i)
        {
            if (!(imgInput >> pixels[i]))
            {
                throw std::runtime_error("Erro ao ler pixel da imagem PPM");
            }
        }

        // Criar matrizes para cada canal RGB
        Matrix matR(height, width);
        Matrix matG(height, width);
        Matrix matB(height, width);
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int idx = (i * width + j) * 3;
                matR.setElement(i, j, pixels[idx + 0] / (double)maxval);
                matG.setElement(i, j, pixels[idx + 1] / (double)maxval);
                matB.setElement(i, j, pixels[idx + 2] / (double)maxval);
            }
        }

        // // Ler kernel
        // std::cout << "Lendo kernel..." << std::endl;
        Matrix kernel = SOBEL_V;
        // std::cout << "Kernel lido:" << std::endl;
        // kernel.print();

        // Realizar detecção de cantos para cada canal
        Matrix resultR = detectEdges(matR);
        Matrix resultG = detectEdges(matG);
        Matrix resultB = detectEdges(matB);

        // Salvar resultado como PPM
        std::string outputFile = (argc > 2) ? argv[2] : "output.ppm";
        std::ofstream output(outputFile);
        if (!output)
            throw std::runtime_error("Não foi possível abrir o arquivo de saída");

        // Escrever cabeçalho PPM com as dimensões ORIGINAIS
        output << "P3\n"
               << width << " " << height << "\n"
               << maxval << "\n";

        // Parâmetro para threshold dos cantos (ajuste conforme necessário)
        double edge_threshold = 0.25; // 0.0 a 1.0 (aumente para detectar menos cantos)

        // Escrever pixels (ajustando para as dimensões originais)
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int r = 0, g = 0, b = 0;
                if (i < resultR.getRows() && j < resultR.getCols())
                {
                    double valR = resultR.getElement(i, j);
                    double valG = resultG.getElement(i, j);
                    double valB = resultB.getElement(i, j);
                    // Aplica threshold: só valores acima do limiar viram "canto"
                    r = (valR > edge_threshold) ? maxval : 0;
                    g = (valG > edge_threshold) ? maxval : 0;
                    b = (valB > edge_threshold) ? maxval : 0;
                }
                output << r << " " << g << " " << b << " ";
            }
            output << "\n";
        }

        output.close();
        std::cout << "Imagem salva em " << outputFile << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Erro: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
