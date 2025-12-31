cmake_minimum_required(VERSION 3.16)
project(NeuralNetworkCpp 
    VERSION 1.0.0
    DESCRIPTION "A lightweight neural network library in pure C++"
    LANGUAGES CXX
)

# ============================================================================
# C++ Standard and Compiler Settings
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support (CLion uses this)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Build Types and Optimization
# ============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall           # Enable all warnings
        -Wextra         # Extra warnings
        -Wpedantic      # Strict ISO C++ compliance
        -Wconversion    # Warn on implicit conversions
        -Wshadow        # Warn when variable shadows another
    )
    
    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -ffast-math)
    endif()
    
    # Debug settings
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0 -fsanitize=address,undefined)
        add_link_options(-fsanitize=address,undefined)
    endif()
elseif(MSVC)
    add_compile_options(/W4 /permissive-)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2)
    endif()
endif()

# ============================================================================
# Library Target
# ============================================================================
# Collect all source files
file(GLOB_RECURSE LIB_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Create the library
add_library(nn_lib STATIC ${LIB_SOURCES})

# Include directories
target_include_directories(nn_lib 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# ============================================================================
# Threading Support (for parallel operations)
# ============================================================================
find_package(Threads REQUIRED)
target_link_libraries(nn_lib PUBLIC Threads::Threads)

# ============================================================================
# Examples
# ============================================================================
option(BUILD_EXAMPLES "Build example programs" ON)

if(BUILD_EXAMPLES)
    # XOR Example
    add_executable(xor_example examples/xor_example.cpp)
    target_link_libraries(xor_example PRIVATE nn_lib)
    
    # MNIST Example
    add_executable(mnist_example examples/mnist_example.cpp)
    target_link_libraries(mnist_example PRIVATE nn_lib)
endif()

# ============================================================================
# Testing
# ============================================================================
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()
    
    # Test executables
    add_executable(test_tensor tests/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE nn_lib)
    add_test(NAME TensorTests COMMAND test_tensor)
    
    add_executable(test_layers tests/test_layers.cpp)
    target_link_libraries(test_layers PRIVATE nn_lib)
    add_test(NAME LayerTests COMMAND test_layers)
    
    add_executable(test_network tests/test_network.cpp)
    target_link_libraries(test_network PRIVATE nn_lib)
    add_test(NAME NetworkTests COMMAND test_network)
endif()

# ============================================================================
# Installation (Optional - for packaging)
# ============================================================================
install(TARGETS nn_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  Neural Network C++ Library")
message(STATUS "========================================")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build tests:    ${BUILD_TESTS}")
message(STATUS "========================================")
message(STATUS "")
