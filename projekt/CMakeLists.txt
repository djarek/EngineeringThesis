cmake_minimum_required(VERSION 2.6)
project(FluidSim)
set(CMAKE_CXX_FLAGS "-std=c++1y -Wall")
add_executable(FluidSim main.cpp)

install(TARGETS FluidSim RUNTIME DESTINATION bin)
target_link_libraries(FluidSim OpenCL)
add_subdirectory(kernels)