# Relatório de Projeto: Raytracer CPU vs. CUDA

## 1. Introdução
Este relatório apresenta uma análise de desempenho de um ray tracer implementado em C++, comparando sua execução em uma CPU (utilizando OpenMP para paralelismo) e em uma GPU (utilizando CUDA). O objetivo é quantificar o *speedup* e a eficiência alcançados com a aceleração por GPU para uma cena e configurações de renderização específicas.

## 2. Configuração do Benchmark
Os benchmarks foram executados com as seguintes configurações de renderização para ambas as versões (CPU e CUDA):

| Parâmetro           | Valor   |
|---------------------|---------|
| Dimensões da Imagem | 80x45   |
| Amostras por Pixel  | 40      |
| Profundidade Máxima | 10      |

## 3. Resultados do Benchmark

### 3.1. Desempenho da Versão CPU
Os dados foram coletados do arquivo `cpu/benchmark_results_cpu.txt` (versão CPU base).

| Métrica                 | Valor (ms) | Valor (FPS) |
|-------------------------|------------|-------------|
| Tempo Médio de Quadro   | 25.85      | 38.68       |
| Tempo Mínimo de Quadro  | 11.65      | 15.56       |
| Tempo Máximo de Quadro  | 64.26      | 85.83       |
| Total de Quadros        | 1208       | N/A         |

### 3.2. Desempenho da Versão CPU (OpenMP)
Os dados foram coletados do arquivo `cpu/benchmark_results_cpu_128_threads.txt` (versão CPU paralelizada com OpenMP).

| Métrica                 | Valor (ms) | Valor (FPS) |
|-------------------------|------------|-------------|
| Tempo Médio de Quadro   | 26.35      | 37.96       |
| Tempo Mínimo de Quadro  | 7.63       | 22.60       |
| Tempo Máximo de Quadro  | 44.25      | 131.15      |
| Total de Quadros        | 1255       | N/A         |

### 3.3. Desempenho da Versão CPU (OpenMP - Dynamic Schedule)
Os dados foram coletados do arquivo `cpu/benchmark_results_cpu_128_threads_dynamic.txt` (versão CPU paralelizada com OpenMP e `schedule(dynamic)`).

| Métrica                 | Valor (ms) | Valor (FPS) |
|-------------------------|------------|-------------|
| Tempo Médio de Quadro   | 24.38      | 41.02       |
| Tempo Mínimo de Quadro  | 7.41       | 6.09        |
| Tempo Máximo de Quadro  | 164.27     | 134.89      |
| Total de Quadros        | 1185       | N/A         |

### 3.4. Desempenho da Versão CUDA
Os dados foram coletados do arquivo `gpu/benchmark_results.txt`.

| Métrica                 | Valor (ms) | Valor (FPS) |
|-------------------------|------------|-------------|
| Tempo Médio de Quadro   | 8.94       | 111.83      |
| Tempo Mínimo de Quadro  | 4.11       | 78.67       |
| Tempo Máximo de Quadro  | 12.71      | 243.17      |
| Total de Quadros        | 998        | N/A         |

## 4. Análise de Desempenho

### 4.1. Cálculo do Speedup
Nesta seção, comparamos o desempenho entre as três versões do ray tracer: CPU Sequencial, CPU com OpenMP (`static` schedule), CPU com OpenMP (`dynamic` schedule) e GPU com CUDA. O *speedup* é calculado como a razão entre o tempo de execução da versão de referência e o da versão comparada.

#### 4.1.1. OpenMP (Static Schedule) vs. Sequencial (CPU)
Esta comparação avalia o ganho de desempenho ao paralelizar o código CPU com OpenMP usando `schedule(static)`.

$$
\text{Speedup} = \frac{\text{Tempo Médio (Sequencial)}}{\text{Tempo Médio (OpenMP Static)}} = \frac{25.85 \, \text{ms}}{26.35 \, \text{ms}} \approx 0.98 \times
$$

Neste caso, a versão com OpenMP (`static` schedule) foi **ligeiramente mais lenta** que a versão sequencial. Isso pode ocorrer devido ao *overhead* de criação e gerenciamento de threads, que pode superar os ganhos de paralelismo em cenas de baixa complexidade ou quando um número excessivo de threads é utilizado para o hardware em questão.

#### 4.1.2. OpenMP (Dynamic Schedule) vs. Sequencial (CPU)
Esta comparação avalia o ganho de desempenho ao paralelizar o código CPU com OpenMP usando `schedule(dynamic)`.
$$ \text{Speedup} = \frac{\text{Tempo Médio (Sequencial)}}{\text{Tempo Médio (OpenMP Dynamic)}} = \frac{25.85 \, \text{ms}}{24.38 \, \text{ms}} \approx 1.06 \times $$
A versão com OpenMP (`dynamic` schedule) foi **ligeiramente mais rápida** que a versão sequencial, indicando que para esta carga de trabalho, o balanceamento de carga dinâmico pode ser benéfico.

#### 4.1.3. OpenMP (Dynamic Schedule) vs. OpenMP (Static Schedule)
Comparando as duas estratégias de agendamento do OpenMP.
$$ \text{Speedup} = \frac{\text{Tempo Médio (OpenMP Static)}}{\text{Tempo Médio (OpenMP Dynamic)}} = \frac{26.35 \, \text{ms}}{24.38 \, \text{ms}} \approx 1.08 \times $$
A versão com `dynamic` schedule é aproximadamente **1.08 vezes mais rápida** que a versão com `static` schedule, sugerindo que o balanceamento de carga dinâmico é mais adequado para esta tarefa.

#### 4.1.4. CUDA vs. Sequencial (CPU)
Aqui, comparamos a aceleração da GPU em relação à implementação mais básica da CPU.

$$
\text{Speedup} = \frac{\text{Tempo Médio (Sequencial)}}{\text{Tempo Médio (CUDA)}} = \frac{25.85 \, \text{ms}}{8.94 \, \text{ms}} \approx 2.89 \times
$$

A versão CUDA é aproximadamente **2.89 vezes mais rápida** que a versão CPU sequencial.

#### 4.1.5. CUDA vs. Melhor OpenMP (CPU - Dynamic Schedule)
Esta é a comparação mais relevante, pois mede o ganho da GPU sobre a melhor versão CPU já paralelizada.

$$
\text{Speedup} = \frac{\text{Tempo Médio (OpenMP Dynamic)}}{\text{Tempo Médio (CUDA)}} = \frac{24.38 \, \text{ms}}{8.94 \, \text{ms}} \approx 2.73 \times
$$

A versão CUDA é aproximadamente **2.73 vezes mais rápida** que a versão CPU paralelizada com OpenMP (`dynamic` schedule).


## 5. Conclusão
A migração do ray tracer para CUDA resultou em um *speedup* notável de aproximadamente **2.73x** em comparação com a melhor versão CPU otimizada com OpenMP (`dynamic` schedule). Este ganho de desempenho valida a abordagem de utilizar a GPU para tarefas computacionalmente intensivas e inerentemente paralelas como o *ray tracing*.

A arquitetura massivamente paralela da GPU, com milhares de CUDA Cores, permite processar muitas trajetórias de raios simultaneamente, superando com folga o paralelismo de poucos núcleos de uma CPU, mesmo quando este é auxiliado por OpenMP. O resultado é uma experiência de renderização significativamente mais rápida e interativa.
