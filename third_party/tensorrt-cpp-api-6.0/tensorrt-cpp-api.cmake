add_library(tensorrt_cpp_api SHARED ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt-cpp-api-6.0/src/engine.cpp)
target_include_directories(tensorrt_cpp_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(tensorrt_cpp_api PUBLIC absl::log ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
target_link_libraries(${PROJECT_NAME} tensorrt_cpp_api)