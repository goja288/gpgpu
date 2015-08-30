cmake_minimum_required(VERSION 3.2)
project(gpgpu_p1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(gpgpu_p1 ${SOURCE_FILES})