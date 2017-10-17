cmake_minimum_required(VERSION 3.7)
project(NeuralClassifier)

set(CMAKE_CXX_STANDARD 11)

FILE(GLOB NeuralNetwork NeuralNetwork/*.cpp)
set(SOURCE_FILES main.cpp ${NeuralNetwork})
add_executable(NeuralClassifier ${SOURCE_FILES})

find_package(PythonLibs 2.7)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(NeuralClassifier ${PYTHON_LIBRARIES})