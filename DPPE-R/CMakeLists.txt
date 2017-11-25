cmake_minimum_required(VERSION 3.0)

project(Parallel_task2)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )
endif()

find_package(MPI REQUIRED)
if(MPI_FOUND)
message(STATUS "MPI INCLUDE_DIRS: " ${MPI_INCLUDE_PATH})
message(STATUS "MPI INCLUDE_LIBS: " ${MPI_C_LIBRARIES})
set(MPI_HOSTS_FILE "${CMAKE_BINARY_DIR}/" CACHE PATH "MPI Hosts file" FORCE)
endif()

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
message(STATUS "OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DUSE_EXPORT_KEYWORD")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(MAIN_SOURCES "src/main.cpp")

include_directories(${MPI_INCLUDE_PATH} mathlib)
add_executable(task2 ${MAIN_SOURCES})
add_subdirectory(mathlib)
target_link_libraries(task2 mathlib ${MPI_C_LIBRARIES})