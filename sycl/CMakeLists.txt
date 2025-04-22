cmake_minimum_required(VERSION 3.25)

project(benchmark LANGUAGES CXX)

set(SOURCES benchmark.cpp)

find_package(IntelSYCL CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} fmt::fmt)
add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${SOURCES})
