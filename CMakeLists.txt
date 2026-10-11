cmake_minimum_required(VERSION 4.0)
project(real_version)

set(CMAKE_CXX_STANDARD 20)

add_executable(real_version main.c)
target_link_libraries(real_version m)