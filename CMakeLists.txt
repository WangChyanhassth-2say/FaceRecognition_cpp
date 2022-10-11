cmake_minimum_required(VERSION 3.0)
project(main)

set(QMAKE_CXXFLAGS "-std=c++11")

find_package(OpenCV REQUIRED)

#4.include头文件目录 
include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/onnx/include
                    ${CMAKE_CURRENT_LIST_DIR}/src)

list(APPEND LINKER_LIBS ${OpenCV_LIBS} 
	${CMAKE_CURRENT_LIST_DIR}/3rdparty/onnx/lib/libonnxruntime.so)

#set(CMAKE_BUILD_TYPE Debug)
AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
add_executable(main ${DIR_SRCS})
target_link_libraries(main ${LINKER_LIBS})

