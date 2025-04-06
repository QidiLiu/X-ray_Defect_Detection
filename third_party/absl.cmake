set(absl_DIR C:/Users/ben/Lib/grpc-install/lib/cmake/absl)
find_package(absl CONFIG REQUIRED)
message(STATUS "[INFO] Using absl-${absl_VERSION}")
target_link_libraries(${PROJECT_NAME} absl::log absl::log_initialize absl::flat_hash_map)
