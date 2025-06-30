#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <cstdlib>

// Функция слияния двух подмассивов.
void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        }
        else {
            temp[k++] = arr[j++];
        }
    }
    while (i <= mid) {
        temp[k++] = arr[i++];
    }
    while (j <= right) {
        temp[k++] = arr[j++];
    }
    for (i = 0; i < k; ++i) {
        arr[left + i] = temp[i];
    }
}

// Рекурсивная функция сортировки слиянием.
void mergeSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

int main(int argc, char* argv[]) {
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
    std::cerr << "Starting with array_size=" << N << "\n";

    // Инициализация генератора случайных чисел.
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Генерация входного массива.
    std::vector<int> arr(N);
    for (int i = 0; i < N; ++i) {
        arr[i] = rand() % 1000;
    }

    // Вычисляем сумму для проверки корректности.
    int sum_orig = 0;
    for (int v : arr) sum_orig += v;

    // Измеряем время сортировки.
    std::cerr << "Running sequential Merge Sort...\n";
    double seq_time = 0.0;
    int sum_seq = 0;
    for (int run = 0; run < 100; ++run) {
        std::vector<int> arr_copy = arr;
        auto t0 = std::chrono::steady_clock::now();
        mergeSort(arr_copy, 0, N - 1);
        auto t1 = std::chrono::steady_clock::now();
        seq_time += std::chrono::duration<double>(t1 - t0).count();
        if (run == 0) {
            for (int v : arr_copy) sum_seq += v;
        }
    }
    seq_time /= 100.0;

    // Вывод результатов.
    std::cout << "\n=== Sequential Merge Sort (average over 100 runs) ===\n";
    std::cout << std::setw(12) << "Array size"
        << std::setw(12) << "Sum"
        << std::setw(20) << "Avg time (s)\n"
        << std::string(44, '-') << "\n";
    std::cout << std::setw(12) << N
        << std::setw(12) << sum_seq
        << std::setw(20) << std::fixed << std::setprecision(6) << seq_time
        << "\n\n";

    // Проверка корректности.
    if (sum_orig != sum_seq) {
        std::cerr << "Error: Sum mismatch! Original: " << sum_orig
            << ", Sequential: " << sum_seq << "\n";
        return 1;
    }

    std::cerr << "Program completed successfully\n";
    return 0;
}