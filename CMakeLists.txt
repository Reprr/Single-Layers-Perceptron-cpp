cmake_minimum_required(VERSION 3.25)
project(linear_model_cpp_impl)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(linear_model_cpp_impl
        main.cpp
        SingleLayerPerceptron.cpp
        SingleLayerPerceptron.h)
