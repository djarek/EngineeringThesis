cmake_minimum_required(VERSION 3.2)
project(FluidSim)
find_package(OpenCL)
set(CMAKE_CXX_FLAGS "-std=c++1y -Wall -pedantic")
add_executable(FluidSim main.cpp simulation.cpp)

install(TARGETS FluidSim RUNTIME DESTINATION bin)
target_link_libraries(FluidSim OpenCL)
add_subdirectory(kernels)