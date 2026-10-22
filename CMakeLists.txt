cmake_minimum_required(VERSION 3.20)
project(real_version LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(real_version main.c)
target_link_libraries(real_version PRIVATE OpenMP::OpenMP_C m)
