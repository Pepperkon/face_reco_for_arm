cmake_minimum_required(VERSION 3.15)
project(SmartCam VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#flags for arm
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mcpu=cortex-a72 -mtune=cortex-a72")

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ---------------------------------------------------------
# c++ engine
# ---------------------------------------------------------
set(CORE_SOURCES
    src/utils.cpp
)

add_library(smartcam_core STATIC ${CORE_SOURCES})
target_link_libraries(smartcam_core ${OpenCV_LIBS})

add_executable(smartcam_run main.cpp)
target_link_libraries(smartcam_run smartcam_core)
