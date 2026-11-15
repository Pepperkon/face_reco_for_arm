cmake_minimum_required(VERSION 3.20)
project(real_version)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(real_version main.c)
target_link_libraries(real_version PRIVATE OpenMP::OpenMP_C m)

# NEON mock test: runs the NEON 1x1 conv code path on any machine and compares with the x86 version
add_executable(test_conv1x1 tests/test_conv1x1.c)
target_compile_definitions(test_conv1x1 PRIVATE NEON_MOCK)
target_include_directories(test_conv1x1 PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_conv1x1 PRIVATE OpenMP::OpenMP_C m)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_compile_options(-mcpu=cortex-a72 -O3 -fopenmp)
    add_link_options(-fopenmp)
endif()