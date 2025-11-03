set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

cmake_minimum_required(VERSION 3.25)

project(benchmark LANGUAGES CXX)

set(SOURCES benchmark.cpp)

find_package(visiongl CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
target_link_libraries(${PROJECT_NAME} PRIVATE visiongl::visiongl fmt::fmt)
