set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

cmake_minimum_required(VERSION 3.25)

project(benchmark LANGUAGES CXX)

set(SOURCES benchmark.cpp)

find_package(AdaptiveCpp CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

set(CMAKE_C_COMPILER acpp)
set(CMAKE_CXX_COMPILER acpp)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} fmt::fmt)
add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${SOURCES})
