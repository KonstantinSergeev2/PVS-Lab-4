#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <cstdlib>

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

     // Ядро CUDA для шага Bitonic Sort.
__global__ void bitonicSortStep(int* arr, int n, int k, int j) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ixj = idx ^ j;
    if (ixj > idx && idx < n && ixj < n) {
        bool up = ((idx & k) == 0);
        int a = arr[idx];
        int b = arr[ixj];
        if ((a > b) == up) {
            arr[idx] = b;
            arr[ixj] = a;
        }
    }
}

int main(int argc, char* argv[]) {
    // Диагностика окружения.
    std::cerr << "Program started on host: " << std::getenv("HOSTNAME") << "\n";
    std::cerr << "CUDA_VISIBLE_DEVICES=" << (std::getenv("CUDA_VISIBLE_DEVICES") ? std::getenv("CUDA_VISIBLE_DEVICES") : "not set") << "\n";
    std::cerr << "LD_LIBRARY_PATH=" << (std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : "not set") << "\n";

    // Проверяем входные аргументы.
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <array_size>\n"
            << "Example: " << argv[0] << " 262144\n";
        return 1;
    }
    int N = std::atoi(argv[1]);
    if (N <= 100000) {
        std::cerr << "Error: array_size must be greater than 100000\n";
        return 1;
    }
    // Bitonic Sort требует, чтобы размер был степенью двойки.
    int pow2N = 1;
    while (pow2N < N) pow2N <<= 1;
    if (pow2N != N) {
        std::cerr << "Error: array_size must be a power of 2 for Bitonic Sort, got " << N << "\n";
        return 1;
    }
    std::cerr << "Starting with array_size=" << N << "\n";

    // Проверяем доступность GPU.
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaGetDeviceCount: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA-capable devices found\n";
        return 1;
    }
    std::cerr << "Found " << deviceCount << " CUDA device(s)\n";
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cerr << "Device " << i << ": " << prop.name
            << ", Compute Capability: " << prop.major << "." << prop.minor
            << ", Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    }
    CUDA_CHECK(cudaSetDevice(0));
    std::cerr << "Selected device 0\n";

    // Инициализация генератора случайных чисел.
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Генерация входного массива.
    std::vector<int> h_array(N);
    for (int i = 0; i < N; ++i) {
        h_array[i] = rand() % 1000;
    }

    // Вычисляем сумму для проверки корректности.
    int sum_orig = 0;
    for (int v : h_array) sum_orig += v;

    // Параллельная сортировка (Bitonic Sort).
    std::cerr << "Running parallel Bitonic Sort...\n";
    int* d_array;
    CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 512;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double par_time = 0.0;
    int sum_par = 0;
    for (int run = 0; run < 100; ++run) {
        CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start));
        for (int k = 2; k <= N; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                bitonicSortStep << <blocks, threadsPerBlock >> > (d_array, N, k, j);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        par_time += ms * 1e-3f;
        if (run == 0) {
            std::vector<int> h_result(N);
            CUDA_CHECK(cudaMemcpy(h_result.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));
            for (int v : h_result) sum_par += v;
        }
    }
    par_time /= 100.0;

    // Вывод результатов.
    std::cout << "\n=== Parallel Bitonic Sort (average over 100 runs) ===\n";
    std::cout << std::setw(12) << "Array size"
        << std::setw(12) << "Sum"
        << std::setw(20) << "Avg time (s)\n"
        << std::string(44, '-') << "\n";
    std::cout << std::setw(12) << N
        << std::setw(12) << sum_par
        << std::setw(20) << std::fixed << std::setprecision(6) << par_time
        << "\n\n";

    // Проверка корректности.
    if (sum_orig != sum_par) {
        std::cerr << "Error: Sum mismatch! Original: " << sum_orig
            << ", Parallel: " << sum_par << "\n";
        return 1;
    }

    CUDA_CHECK(cudaFree(d_array));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cerr << "Program completed successfully\n";
    return 0;
}