#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <ctime>

// ������ ��� �������� ������ CUDA.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << #call << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ���� CUDA ��� �������� �����.
__global__ void sumReduction(const int* input, int* output, int size) {
    // ����������� ������ ��� ��������.
    extern __shared__ int sdata[];

    // �������������� ������ � �����.
    int tid = threadIdx.x; // ������ ������ � �����.
    int idx = blockIdx.x * blockDim.x + tid; // ���������� ������ ������.

    // �������� ������ � ����������� ������.
    sdata[tid] = (idx < size) ? input[idx] : 0;
    __syncthreads();

    // ������������ �������� � �����.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // ���������� ���������� �����.
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char* argv[]) {
    // ��������� ������� ���������.
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <array_size>\n"
            << "Example: " << argv[0] << " 200000\n";
        return 1;
    }
    int N = std::atoi(argv[1]); // ���������� ��������� �������.
    if (N <= 100000) {
        std::cerr << "Error: array_size must be greater than 100000\n";
        return 1;
    }

    // ������������� ���������� ��������� �����.
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // ��������� �������� ������� �� CPU.
    std::vector<int> h_array(N);
    for (int i = 0; i < N; ++i) {
        h_array[i] = rand() % 1000;
    }

    // ���������������� ������: ��������� ����� ���� ���.
    int sum_seq = 0;
    for (int v : h_array) sum_seq += v;

    // �������� ����� ���������������� ������.
    double seq_time = 0.0;
    for (int run = 0; run < 100; ++run) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int s = 0; // ������ ���� ��� ��������� �������..
        for (int v : h_array) s += v;
        auto t1 = std::chrono::high_resolution_clock::now();
        seq_time += std::chrono::duration<double>(t1 - t0).count();
    }
    seq_time /= 100.0;

    // �������� ������ �� GPU.
    int* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // ������ ��������� ���������� ������� � �����.
    std::vector<int> tpb_list = { 256, 512, 1024 };

    // ��������� ������� ������������� ����������
    std::cout << "\n=== Parallel Execution (average over 100 runs) ===\n";
    std::cout << std::setw(12) << "Array size"
        << std::setw(12) << "Blocks"
        << std::setw(18) << "Threads/block"
        << std::setw(15) << "Total threads"
        << std::setw(12) << "Sum"
        << std::setw(20) << "Avg time (s)\n"
        << std::string(80, '-') << "\n";

    for (int tpb : tpb_list) {
        // �������� �� ������������ ���������� �������.
        if (tpb > 1024) {
            std::cout << "Skipping threads_per_block=" << tpb << " (exceeds max threads per block)\n";
            continue;
        }

        int blocks = (N + tpb - 1) / tpb; // ����� ������.
        int* d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(int)));

        // ������� ��� ��������� �������.
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double par_time = 0.0;
        int sum_par = 0;

        for (int run = 0; run < 100; ++run) {
            CUDA_CHECK(cudaEventRecord(start));

            // ������ ���� ��������.
            sumReduction << <blocks, tpb, tpb * sizeof(int) >> > (d_input, d_partial, N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // ����������� ��������.
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
            par_time += ms * 1e-3f; // ��������� � �������.

            CUDA_CHECK(cudaMemcpy(&sum_par, d_partial, sizeof(int), cudaMemcpyDeviceToHost));
        }
        par_time /= 100.0;

        // ����� ����������� ��� �������� ��������.
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

    // ����� ���������� ����������������� ����������.
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