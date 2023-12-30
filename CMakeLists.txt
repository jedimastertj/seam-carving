cmake_minimum_required(VERSION 3.5.0)
project(LiquidRescaling)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(LiquidRescaling main.cpp)

target_link_libraries(LiquidRescaling ${OpenCV_LIBS})
