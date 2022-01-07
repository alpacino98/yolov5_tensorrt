cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/src/yololayer.cu ${PROJECT_SOURCE_DIR}/src/maskRCNNKernels.cu ${PROJECT_SOURCE_DIR}/src/resizeNearestPlugin.cpp)
target_link_libraries(myplugins nvinfer cudart)

file(GLOB_RECURSE yolov5_sources
 ${PROJECT_SOURCE_DIR}/src/*.cpp)

#file(GLOB SRC_FILES "./src/resizeNearestPlugin.cpp")
add_library(resizeNearest SHARED ${PROJECT_SOURCE_DIR}/src/resizeNearestPlugin.cpp)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

#include_directories(resizeNearestPlugin)
#add_subdirectory(resizeNearestPlugin)

add_executable(yolov5 ${yolov5_sources})
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

