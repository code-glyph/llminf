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

# engine library    
add_library(engine cpp/engine/weights.cpp cpp/engine/model.cpp cpp/engine/generate.cpp)
target_link_libraries(engine ${TORCH_LIBRARIES})
target_include_directories(engine PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(engine PRIVATE ${TORCH_CXX_FLAGS})

# bench binary
add_executable(bench cpp/bench/main.cpp)
target_link_libraries(bench engine ${TORCH_LIBRARIES})
target_compile_options(bench PRIVATE ${TORCH_CXX_FLAGS})