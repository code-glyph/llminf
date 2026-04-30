cmake_minimum_required(VERSION 3.18)
project(llminf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# libtorch
list(APPEND CMAKE_PREFIX_PATH "/home/dbd5616/miniconda3/lib/python3.13/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)

# includes
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# engine library (empty for now, we'll add sources as we go)
add_library(engine INTERFACE)
target_include_directories(engine INTERFACE ${CMAKE_SOURCE_DIR}/include)

# bench binary
add_executable(bench cpp/bench/main.cpp)
target_link_libraries(bench ${TORCH_LIBRARIES})
target_compile_options(bench PRIVATE ${TORCH_CXX_FLAGS})