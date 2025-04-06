cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME infer-electrode-mdcnet-trt)
project(${PROJECT_NAME} C CXX)
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Third party libraries
include("${CMAKE_CURRENT_SOURCE_DIR}/third_party/absl.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/third_party/TensorRT.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenCV.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt-cpp-api-6.0/tensorrt-cpp-api.cmake")
