#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <ctime>

// Макрос для проверки ошибок CUDA.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << #call << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Ядро CUDA для редукции суммы.
__global__ void sumReduction(const int* input, int* output, int size) {
    // Разделяемая память для редукции.
    extern __shared__ int sdata[];

    // Идентификаторы потока и блока.
    int tid = threadIdx.x; // Индекс потока в блоке.
    int idx = blockIdx.x * blockDim.x + tid; // Глобальный индекс потока.

    // Загрузка данных в разделяемую память.
    sdata[tid] = (idx < size) ? input[idx] : 0;
    __syncthreads();

    // Параллельная редукция в блоке.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Сохранение результата блока.
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char* argv[]) {
    // Проверяем входные аргументы.
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <array_size>\n"
            << "Example: " << argv[0] << " 200000\n";
        return 1;
    }
    int N = std::atoi(argv[1]); // Количество элементов массива.
    if (N <= 100000) {
        std::cerr << "Error: array_size must be greater than 100000\n";
        return 1;
    }

    // Инициализация генератора случайных чисел.
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Генерация входного массива на CPU.
    std::vector<int> h_array(N);
    for (int i = 0; i < N; ++i) {
        h_array[i] = rand() % 1000;
    }

    // Последовательная версия: вычисляем сумму один раз.
    int sum_seq = 0;
    for (int v : h_array) sum_seq += v;

    // Измеряем время последовательной версии.
    double seq_time = 0.0;
    for (int run = 0; run < 100; ++run) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int s = 0; // Пустой цикл для измерения времени..
        for (int v : h_array) s += v;
        auto t1 = std::chrono::high_resolution_clock::now();
        seq_time += std::chrono::duration<double>(t1 - t0).count();
    }
    seq_time /= 100.0;

    // Копируем данные на GPU.
    int* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Список вариантов количества потоков в блоке.
    std::vector<int> tpb_list = { 256, 512, 1024 };

    // Заголовок таблицы параллельного выполнения
    std::cout << "\n=== Parallel Execution (average over 100 runs) ===\n";
    std::cout << std::setw(12) << "Array size"
        << std::setw(12) << "Blocks"
        << std::setw(18) << "Threads/block"
        << std::setw(15) << "Total threads"
        << std::setw(12) << "Sum"
        << std::setw(20) << "Avg time (s)\n"
        << std::string(80, '-') << "\n";

    for (int tpb : tpb_list) {
        // Проверка на максимальное количество потоков.
        if (tpb > 1024) {
            std::cout << "Skipping threads_per_block=" << tpb << " (exceeds max threads per block)\n";
            continue;
        }

        int blocks = (N + tpb - 1) / tpb; // Число блоков.
        int* d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(int)));

        // События для измерения времени.
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double par_time = 0.0;
        int sum_par = 0;

        for (int run = 0; run < 100; ++run) {
            CUDA_CHECK(cudaEventRecord(start));

            // Первый этап редукции.
            sumReduction << <blocks, tpb, tpb * sizeof(int) >> > (d_input, d_partial, N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Итеративная редукция.
            int rem = blocks;
            int* curr = d_partial;
            while (rem > 1) {
                int nb = (rem + tpb - 1) / tpb;
                sumReduction << <nb, tpb, tpb * sizeof(int) >> > (curr, curr, rem);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                rem = nb;
            }

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            par_time += ms * 1e-3f; // Переводим в секунды.

            CUDA_CHECK(cudaMemcpy(&sum_par, d_partial, sizeof(int), cudaMemcpyDeviceToHost));
        }
        par_time /= 100.0;

        // Вывод результатов для текущего варианта.
        std::cout << std::setw(12) << N
            << std::setw(12) << blocks
            << std::setw(18) << tpb
            << std::setw(15) << (blocks * tpb)
            << std::setw(12) << sum_par
            << std::setw(20) << std::fixed << std::setprecision(6) << par_time
            << "\n";

        CUDA_CHECK(cudaFree(d_partial));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // Вывод результата последовательного выполнения.
    std::cout << "\n=== Sequential Execution (average over 100 runs) ===\n";
    std::cout << std::setw(12) << "Array size"
        << std::setw(12) << "Sum"
        << std::setw(20) << "Avg time (s)\n"
        << std::string(44, '-') << "\n";
    std::cout << std::setw(12) << N
        << std::setw(12) << sum_seq
        << std::setw(20) << std::fixed << std::setprecision(6) << seq_time
        << "\n\n";

    CUDA_CHECK(cudaFree(d_input));
    return 0;
}