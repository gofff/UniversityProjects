cmake_minimum_required(VERSION 2.6)

project(Parallel_task2)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )
endif()
set(CMAKE_CXX_FLAGS "-fopenmp -std=c++0x")

find_package(MPI REQUIRED)

set(MAIN_SOURCES "src/main.cpp")

include_directories(${MPI_INCLUDE_PATH} mathlib)
add_executable(task2 ${MAIN_SOURCES})
add_subdirectory(mathlib)
target_link_libraries(task2 mathlib ${MPI_C_LIBRARIES})