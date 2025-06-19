#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <fftw3.h>
#include <cmath>

class Matrix {
private:
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

public:
    Matrix() {
        std::cin >> rows >> cols;
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument("Dimensões inválidas da matriz");
        }
        data.resize(rows, std::vector<double>(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; j++) {
                if (!(std::cin >> data[i][j])) {
                    throw std::runtime_error("Erro ao ler elementos da matriz");
                }
            }
        }
    }

    // Construtor com dados direto
    Matrix(std::vector<std::vector<double>> d) : data(d), rows(d.size()), cols(d[0].size()) {}

    // Construtor para leitura de um canal de imagem PPM P3
    Matrix(std::istream& input, int r, int c, int channel, int maxval) : rows(r), cols(c) {
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument("Dimensões inválidas da matriz");
        }
        data.resize(rows);
        for (int i = 0; i < rows; ++i) {
            data[i].resize(cols);
        }
        // Ler pixels (P3: cada linha tem R G B)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int r, g, b;
                if (!(input >> r >> g >> b)) {
                    throw std::runtime_error("Erro ao ler pixel da imagem PPM");
                }
                // Normalizar para [0, 1]
                if (channel == 0) data[i][j] = r / (double)maxval; // R
                else if (channel == 1) data[i][j] = g / (double)maxval; // G
                else if (channel == 2) data[i][j] = b / (double)maxval; // B
            }
        }
    }

    // Construtor com dimensões específicas (para criar matriz de saída)
    Matrix(int r, int c) : rows(r), cols(c) {
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument("Dimensões inválidas da matriz");
        }
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    // Método para exibir a matriz
    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Salvar canal como parte de uma imagem PPM (um canal por vez)
    void saveAsPPMChannel(std::ostream& output, int maxval, int channel) const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int val = std::round(data[i][j] * maxval);
                val = std::max(0, std::min(maxval, val)); // Clamp
                if (channel == 0) output << val << " 0 0 "; // R
                else if (channel == 1) output << "0 " << val << " 0 "; // G
                else if (channel == 2) output << "0 0 " << val << " "; // B
            }
            output << "\n";
        }
    }

    int getRows() const { return rows; }

    int getCols() const { return cols; }

    double getElement(int i, int j) const {
        if (i < 0 || i >= rows || j < 0 || j >= cols) {
            throw std::out_of_range("Índice fora dos limites");
        }
        return data[i][j];
    }

    void setElement(int i, int j, double value) {
        if (i < 0 || i >= rows || j < 0 || j >= cols) {
            throw std::out_of_range("Índice fora dos limites");
        }
        data[i][j] = value;
    }

    Matrix convolve2D(const Matrix& kernel) const {
        int N = rows;
        int M = cols;
        int K = kernel.getRows();
        int L = kernel.getCols();

        // Tamanho da saída da convolução
        int outRows = N + K - 1;
        int outCols = M + L - 1;

        // Tamanho para FFT (próxima potência de 2 para eficiência)
        int fftRows = std::pow(2, std::ceil(std::log2(outRows)));
        int fftCols = std::pow(2, std::ceil(std::log2(outCols)));

        // Alocar memória para FFT
        fftw_complex *in_mat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftRows * fftCols);
        fftw_complex *in_ker = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftRows * fftCols);
        fftw_complex *out_mat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftRows * fftCols);
        fftw_complex *out_ker = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftRows * fftCols);
        fftw_complex *out_result = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftRows * fftCols);
        fftw_complex *in_result = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftRows * fftCols);

        // Inicializar com zeros
        for (int i = 0; i < fftRows * fftCols; ++i) {
            in_mat[i][0] = in_mat[i][1] = 0.0;
            in_ker[i][0] = in_ker[i][1] = 0.0;
        }

        // Preencher matriz e kernel com zero-padding
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                in_mat[i * fftCols + j][0] = data[i][j];
            }
        }
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < L; ++j) {
                in_ker[i * fftCols + j][0] = kernel.getElement(i, j);
            }
        }

        // Criar planos para FFT 2D
        fftw_plan plan_mat = fftw_plan_dft_2d(fftRows, fftCols, in_mat, out_mat, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan plan_ker = fftw_plan_dft_2d(fftRows, fftCols, in_ker, out_ker, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan plan_result = fftw_plan_dft_2d(fftRows, fftCols, in_result, out_result, FFTW_BACKWARD, FFTW_ESTIMATE);

        // Executar FFT
        fftw_execute(plan_mat);
        fftw_execute(plan_ker);

        // Multiplicação ponto a ponto
        for (int i = 0; i < fftRows * fftCols; ++i) {
            in_result[i][0] = out_mat[i][0] * out_ker[i][0] - out_mat[i][1] * out_ker[i][1]; // Real
            in_result[i][1] = out_mat[i][0] * out_ker[i][1] + out_mat[i][1] * out_ker[i][0]; // Imaginário
        }

        // Executar IFFT
        fftw_execute(plan_result);

        // Criar matriz de saída
        Matrix result(outRows, outCols);
        for (int i = 0; i < outRows; ++i) {
            for (int j = 0; j < outCols; ++j) {
                result.setElement(i, j, out_result[i * fftCols + j][0] / (fftRows * fftCols));
            }
        }

        // Liberar memória
        fftw_destroy_plan(plan_mat);
        fftw_destroy_plan(plan_ker);
        fftw_destroy_plan(plan_result);
        fftw_free(in_mat); fftw_free(in_ker); fftw_free(out_mat);
        fftw_free(out_ker); fftw_free(out_result); fftw_free(in_result);

        return result;
    }

};

#endif // !MATRIX_H
