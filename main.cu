#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda.h>

__global__ void blelloch_up_sweep(int* array, int tamanho, int* trabalho, int* passos)
{
    int passo = 1;
    while (passo < tamanho)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        int offset = 2 * passo * index;
        if (offset + 2 * passo - 1 < tamanho)
        {
            array[offset + 2 * passo - 1] += array[offset + passo - 1];
            atomicAdd(trabalho, 1); // Contabiliza o trabalho da soma
        }
        if (index == 0) atomicAdd(passos, 1); // Contabiliza um passo por iteração
        passo *= 2;
        __syncthreads();
    }
}

__global__ void blelloch_down_sweep(int* array, int tamanho, int* trabalho, int* passos)
{
    array[tamanho - 1] = 0;
    int passo = tamanho / 2;
    while (passo > 0)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        int offset = 2 * passo * index;
        if (offset + 2 * passo - 1 < tamanho)
        {
            int temp = array[offset + passo - 1];
            array[offset + passo - 1] = array[offset + 2 * passo - 1];
            array[offset + 2 * passo - 1] += temp;
            atomicAdd(trabalho, 1); // Contabiliza o trabalho da troca
        }
        if (index == 0) atomicAdd(passos, 1); // Contabiliza um passo por iteração
        passo /= 2;
        __syncthreads();
    }
}

void medir_tempo_blelloch(int tamanho)
{
    int* h_array = new int[tamanho];
    std::fill_n(h_array, tamanho, 1);

    int* d_array;
    cudaMalloc((void**)&d_array, tamanho * sizeof(int));
    cudaMemcpy(d_array, h_array, tamanho * sizeof(int), cudaMemcpyHostToDevice);

    int h_trabalho = 0, h_passos = 0;
    int *d_trabalho, *d_passos;
    cudaMalloc((void**)&d_trabalho, sizeof(int));
    cudaMalloc((void**)&d_passos, sizeof(int));
    cudaMemcpy(d_trabalho, &h_trabalho, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_passos, &h_passos, sizeof(int), cudaMemcpyHostToDevice);

    auto inicio = std::chrono::high_resolution_clock::now();
    int blocos = (tamanho + 255) / 256;
    blelloch_up_sweep<<<blocos, 256>>>(d_array, tamanho, d_trabalho, d_passos);
    cudaDeviceSynchronize();
    blelloch_down_sweep<<<blocos, 256>>>(d_array, tamanho, d_trabalho, d_passos);
    cudaDeviceSynchronize();
    auto fim = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duracao = fim - inicio;

    cudaMemcpy(&h_trabalho, d_trabalho, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_passos, d_passos, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Tamanho do array: " << tamanho
        << " | Tempo (Blelloch): " << duracao.count() << " segundos"
        << " | Trabalho: " << h_trabalho
        << " | Passos: " << h_passos << std::endl;

    cudaFree(d_array);
    cudaFree(d_trabalho);
    cudaFree(d_passos);
    delete[] h_array;
}

int main()
{
    int tamanhos[] = {100, 1000, 10000, 100000, 1000000, 10000000};
    for (int tamanho : tamanhos)
    {
        medir_tempo_blelloch(tamanho);
    }
    return 0;
}
