cmake_minimum_required(VERSION 2.8)

project(editVideo)

set( CMAKE_CXX_FLAGS "-std=c++11")
set( CMAKE_BUILD_TYPE "Debug")

find_package(OpenCV 3.1 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(editVideo editVideo.cpp)

target_link_libraries(editVideo ${OpenCV_LIBS})


