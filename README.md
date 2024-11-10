# Blelloch em CUDA

Implementação experimental da soma de prefixos exclusiva (exclusive scan) de Blelloch em CUDA. Em um scan exclusivo, cada posição recebe a soma dos elementos anteriores. Por exemplo, `[1, 1, 1, 1]` se torna `[0, 1, 2, 3]`.

[main.cu](main.cu) contém a redução em árvore (`up_sweep`), a propagação dos prefixos (`down_sweep`) e o programa de medição. [CMakeLists.txt](CMakeLists.txt) define o alvo `Blelloch`.

## Como executar

Use CMake 3.29 ou superior, uma GPU NVIDIA compatível e uma instalação do CUDA Toolkit com suporte ao padrão CUDA C++20 e ao compilador anfitrião.

```bash
cmake -S . -B build
cmake --build build --config Release
```

Execute o alvo `Blelloch` gerado. Com Visual Studio, procure `build/Release/Blelloch.exe`. Com CMake de configuração única, procure `build/Blelloch` ou `build/Blelloch.exe`.

O programa preenche vetores com `1` e processa os tamanhos de 100 a 10.000.000 elementos declarados em `main`. A árvore recebe zeros até a próxima potência de dois e o programa copia de volta os prefixos dos elementos originais. A saída mostra o tempo, o último prefixo e o número de etapas da árvore. Em um vetor de 100 uns, o último prefixo é 99.
