cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

find_package(Torch REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)


add_executable(RKKY_test RKKY_test.cxx)

include_directories(RKKY_test "${CMAKE_SOURCE_DIR}")

target_link_libraries(RKKY_test "${TORCH_LIBRARIES}")
target_link_libraries(RKKY_test Eigen3::Eigen)
target_link_libraries(RKKY_test OpenMP::OpenMP_CXX)

add_executable(ML_test ML_test.cxx)

include_directories(ML_test "${CMAKE_SOURCE_DIR}")

target_link_libraries(ML_test "${TORCH_LIBRARIES}")
target_link_libraries(ML_test Eigen3::Eigen)
target_link_libraries(ML_test OpenMP::OpenMP_CXX)


set_property(TARGET RKKY_test PROPERTY CXX_STANDARD 17)
set_property(TARGET ML_test PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -fopenmp -march=native")
