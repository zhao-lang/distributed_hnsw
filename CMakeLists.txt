cmake_minimum_required(VERSION 3.11)

# set the project name
project(Distributed_HNSW LANGUAGES CXX VERSION 0.1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-mavx2 -mfma ")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# grpc++
include(FetchContent)
FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc.git
  GIT_TAG        v1.26.0)
FetchContent_MakeAvailable(gRPC)

# Since FetchContent uses add_subdirectory under the hood, we can use
  # the grpc targets directly from this build.
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP_UNSECURE grpc++_unsecure)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

# Proto file
get_filename_component(proto "${PROJECT_SOURCE_DIR}/protos/distributed_hnsw.proto" ABSOLUTE)
get_filename_component(proto_path "${proto}" PATH)

# Generated sources
set(proto_srcs "${PROJECT_SOURCE_DIR}/include/pb/distributed_hnsw.pb.cc")
set(proto_hdrs "${PROJECT_SOURCE_DIR}/include/pb/distributed_hnsw.pb.h")
set(grpc_srcs "${PROJECT_SOURCE_DIR}/include/pb/distributed_hnsw.grpc.pb.cc")
set(grpc_hdrs "${PROJECT_SOURCE_DIR}/include/pb/distributed_hnsw.grpc.pb.h")
add_custom_command(
      OUTPUT "${proto_srcs}" "${proto_hdrs}" "${grpc_srcs}" "${grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${PROJECT_SOURCE_DIR}/include/pb"
        --cpp_out "${PROJECT_SOURCE_DIR}/include/pb"
        -I "${proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${proto}"
      DEPENDS "${proto}")

add_subdirectory("include")

# example executable
add_executable(dhnsw_examples examples/main.cpp)

target_link_libraries(dhnsw_examples PUBLIC hnsw)

target_include_directories(dhnsw_examples PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/include"
                          )

# server executable
add_executable(server "src/server/main.cpp" "src/server/core/core.cpp" "src/server/cluster/cluster.cpp"
               ${proto_srcs} ${grpc_srcs})

target_link_libraries(server PUBLIC hnsw pb grpc++)

target_include_directories(server PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/include"
                          )


                          
include(CTest)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    include(Catch)

    # tests executable
    add_executable(hnsw_tests tests/main.cpp tests/hnsw_tests.cpp)

    target_link_libraries(hnsw_tests PUBLIC hnsw)

    target_include_directories(hnsw_tests PUBLIC
                              "${PROJECT_BINARY_DIR}"
                              "${PROJECT_SOURCE_DIR}/include"
                              )

    # test discovery
    catch_discover_tests(hnsw_tests)

    add_custom_command(TARGET hnsw_tests
                    POST_BUILD
                    COMMAND ctest -C $<CONFIGURATION> --output-on-failure)
endif()