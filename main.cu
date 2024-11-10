#include <bit>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

__global__ void up_sweep(int* array, int n, int passo) {
    unsigned par = blockIdx.x * blockDim.x + threadIdx.x;
    if (par < static_cast<unsigned>(n / (2 * passo))) {
        int direita = (par + 1) * 2 * passo - 1;
        array[direita] += array[direita - passo];
    }
}

__global__ void down_sweep(int* array, int n, int passo) {
    unsigned par = blockIdx.x * blockDim.x + threadIdx.x;
    if (par < static_cast<unsigned>(n / (2 * passo))) {
        int direita = (par + 1) * 2 * passo - 1;
        int temp = array[direita - passo];
        array[direita - passo] = array[direita];
        array[direita] += temp;
    }
}

void verificarCuda(cudaError_t status, const char* operacao) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operacao) + ": " + cudaGetErrorString(status));
    }
}

void blelloch_scan(int* array, int n) {
    if (n <= 0) return;

    int tamanhoArvore = static_cast<int>(std::bit_ceil(static_cast<unsigned>(n)));
    int* d_array = nullptr;
    verificarCuda(cudaMalloc(&d_array, tamanhoArvore * sizeof(int)), "alocar vetor CUDA");
    std::unique_ptr<int, decltype(&cudaFree)> buffer(d_array, cudaFree);
    verificarCuda(cudaMemset(d_array, 0, tamanhoArvore * sizeof(int)), "inicializar vetor CUDA");
    verificarCuda(cudaMemcpy(d_array, array, n * sizeof(int), cudaMemcpyHostToDevice), "enviar vetor");

    constexpr int threads = 256;
    for (int passo = 1; passo < tamanhoArvore; passo *= 2) {
        int pares = tamanhoArvore / (passo * 2);
        int blocos = (pares + threads - 1) / threads;
        up_sweep<<<blocos, threads>>>(d_array, tamanhoArvore, passo);
        verificarCuda(cudaGetLastError(), "lançar up-sweep");
    }

    verificarCuda(cudaMemset(&d_array[tamanhoArvore - 1], 0, sizeof(int)), "zerar raiz da árvore");

    for (int passo = tamanhoArvore / 2; passo > 0; passo /= 2) {
        int pares = tamanhoArvore / (passo * 2);
        int blocos = (pares + threads - 1) / threads;
        down_sweep<<<blocos, threads>>>(d_array, tamanhoArvore, passo);
        verificarCuda(cudaGetLastError(), "lançar down-sweep");
    }

    verificarCuda(cudaMemcpy(array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost), "receber prefixos");
}

int main() try {
    const int tamanhos[] = {100, 1000, 10000, 100000, 1000000, 10000000};

    for (int n : tamanhos) {
        std::vector<int> array(n, 1);

        auto start = std::chrono::high_resolution_clock::now();
        blelloch_scan(array.data(), n);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        int passos = 2 * std::bit_width(static_cast<unsigned>(n - 1));
        std::cout << "Tamanho do array: " << n
                  << " | Tempo (Blelloch): " << diff.count() << " segundos"
                  << " | Último prefixo: " << array.back()
                  << " | Passos: " << passos << std::endl;
    }
    return 0;
} catch (const std::exception& erro) {
    std::cerr << erro.what() << std::endl;
    return 1;
}
