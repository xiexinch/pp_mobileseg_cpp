cmake_minimum_required(VERSION 3.26)
project(pp_mobielseg)
add_executable(pp_mobielseg inferencer.cpp)

find_package(OpenCV)
include_directories( ${OpenCV_INCLUDE_DIRS} )

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -pthread")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)
target_include_directories(pp_mobielseg PRIVATE
    ${CMAKE_SOURCE_DIR}/3rdparty_libs/onnxruntime/include
)

target_link_libraries(pp_mobielseg PRIVATE
    ${CMAKE_SOURCE_DIR}/3rdparty_libs/onnxruntime/lib/libonnxruntime.1.15.1.dylib
    ${OpenCV_LIBS}
)
