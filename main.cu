#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

__global__ void up_sweep(int* array, int n, int passo) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * passo * 2;
    if (idx + passo < n) {
        array[idx + passo * 2 - 1] += array[idx + passo - 1];
    }
}

__global__ void down_sweep(int* array, int n, int passo) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * passo * 2;
    if (idx + passo < n) {
        int temp = array[idx + passo - 1];
        array[idx + passo - 1] = array[idx + passo * 2 - 1];
        array[idx + passo * 2 - 1] += temp;
    }
}

void blelloch_scan(int* array, int n) {
    int* d_array;
    cudaMalloc(&d_array, n * sizeof(int));
    cudaMemcpy(d_array, array, n * sizeof(int), cudaMemcpyHostToDevice);

    int passo = 1;
    while (passo < n) {
        int numThreads = (n / (passo * 2) + 255) / 256;
        up_sweep<<<numThreads, 256>>>(d_array, n, passo);
        passo *= 2;
        cudaDeviceSynchronize();
    }

    cudaMemset(&d_array[n - 1], 0, sizeof(int)); // Reset para down-sweep

    passo /= 2;
    while (passo > 0) {
        int numThreads = (n / (passo * 2) + 255) / 256;
        down_sweep<<<numThreads, 256>>>(d_array, n, passo);
        passo /= 2;
        cudaDeviceSynchronize();
    }

    cudaMemcpy(array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

int main() {
    int tamanhos[] = {100, 1000, 10000, 100000, 1000000, 10000000};

    for (int i = 0; i < 6; ++i) {
        int n = tamanhos[i];
        int* array = new int[n];
        for (int j = 0; j < n; ++j) array[j] = 1;

        auto start = std::chrono::high_resolution_clock::now();

        blelloch_scan(array, n);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        int soma = array[n - 1];
        int passos = log2(n) * 2;  // Estimativa para o número total de passos

        std::cout << "Tamanho do array: " << n
                  << " | Tempo (Blelloch): " << diff.count() << " segundos"
                  << " | Trabalho: " << soma
                  << " | Passos: " << passos << std::endl;

        delete[] array;
    }
    return 0;
}
