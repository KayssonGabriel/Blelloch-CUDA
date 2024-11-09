cmake_minimum_required(VERSION 3.29)
project(Blelloch CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Blelloch main.cu)

set_target_properties(Blelloch PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
