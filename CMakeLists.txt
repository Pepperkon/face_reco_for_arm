cmake_minimum_required(VERSION 3.20)
project(real_version)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(real_version main.c)
target_link_libraries(real_version PRIVATE OpenMP::OpenMP_C m)