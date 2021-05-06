cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

find_package(Torch REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)


add_executable(test RKKY_test.cxx)

target_link_libraries(test "${TORCH_LIBRARIES}")
target_link_libraries(test Eigen3::Eigen)
target_link_libraries(test OpenMP::OpenMP_CXX)

set_property(TARGET test PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -O3 -march=native")