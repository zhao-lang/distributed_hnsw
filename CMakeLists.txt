cmake_minimum_required(VERSION 3.11)

# set the project name
project(Distributed_HNSW VERSION 0.1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(include)

# example executable
add_executable(example examples/main.cpp)

target_link_libraries(example PUBLIC hnsw)

target_include_directories(example PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/include"
                          )

# tests executable
add_executable(tests tests/main.cpp tests/hnsw_tests.cpp)

target_link_libraries(tests PUBLIC hnsw)

target_include_directories(tests PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/include"
                          )
