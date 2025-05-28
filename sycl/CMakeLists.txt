set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O3")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    endif()
endif()

get_filename_component(CXX_COMPILER_NAME ${CMAKE_CXX_COMPILER} NAME)
if(NOT CXX_COMPILER_NAME STREQUAL "icpx" AND
   NOT CXX_COMPILER_NAME STREQUAL "icx-cl" AND
   NOT CXX_COMPILER_NAME STREQUAL "acpp")
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER}")
endif()

cmake_minimum_required(VERSION 3.25)

project(benchmark LANGUAGES CXX)

set(SOURCES benchmark.cpp)

if (CXX_COMPILER_NAME STREQUAL "acpp")
    find_package(AdaptiveCpp CONFIG REQUIRED)
else()
    find_package(IntelSYCL CONFIG REQUIRED)
endif()
find_package(OpenCV CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} fmt::fmt)
add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${SOURCES})