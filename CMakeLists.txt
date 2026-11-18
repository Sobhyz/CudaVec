
cmake_minimum_required(VERSION 3.18)

project(CudaVec
    VERSION 0.1.0
    LANGUAGES CXX
)

# C++ configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Core library
add_library(cudavec_core
    src/vector_dataset.cpp
    src/similarity.cpp
    src/utility.cpp
)

# Public headers
target_include_directories(cudavec_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cudavec_core PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# VecDB unit tests
add_executable(test_vec_db
    tests/test_vec_db.cpp
)

target_link_libraries(test_vec_db
    PRIVATE cudavec_core
)

add_executable(test_similarity
    tests/test_similarity.cpp
)

target_link_libraries(test_similarity
    PRIVATE cudavec_core
)

# Enable CTest
include(CTest)

if(BUILD_TESTING)
    add_test(
        NAME VecDBTests
        COMMAND test_vec_db
    )
endif()

if(BUILD_TESTING)
    add_test(
        NAME SimilarityTests
        COMMAND test_similarity
    )
endif()
