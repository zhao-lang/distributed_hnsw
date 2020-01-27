cmake_minimum_required(VERSION 3.11)

# set the project name
project(Distributed_HNSW LANGUAGES CXX VERSION 0.1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

include(FetchContent)

# grpc++
FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.26.0
)
FetchContent_MakeAvailable(gRPC)

add_subdirectory(include)

# example executable
add_executable(dhnsw_examples examples/main.cpp)

target_link_libraries(dhnsw_examples PUBLIC hnsw grpc++)

target_include_directories(dhnsw_examples PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/include"
                          )

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)

    include(CTest)
    include(Catch)

    # tests executable
    add_executable(tests tests/main.cpp tests/hnsw_tests.cpp)

    target_link_libraries(tests PUBLIC hnsw)

    target_include_directories(tests PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/include"
                            )

    # test discovery
    catch_discover_tests(tests)

    add_custom_command(TARGET tests
                    POST_BUILD
                    COMMAND ctest -C $<CONFIGURATION> --output-on-failure)
endif()