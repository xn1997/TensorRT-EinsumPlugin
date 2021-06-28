cmake_minimum_required(VERSION 3.16)

project(TensorRT)

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -o0 -Wall -g -ggdb")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_ROOT "/usr/local/opencv4.4.0/lib/cmake/opencv4")
set(OpenCV_DIR "/usr/local/opencv4.4.0/lib/cmake/opencv4")
find_package(OpenCV  REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_LIBRARIES(${OpenCV_LIBS})

add_subdirectory(einsum einsum)
LINK_LIBRARIES(einsumlib)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR})  # 设定h文件路径
add_subdirectory(function function)
LINK_LIBRARIES(functionlib)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#add_executable(infer_test "infer_test.cpp")
#add_executable(testNet "testNet.cpp")
#add_executable(mnist "mnist.cpp")
#add_executable(resnet50 "resnet50.cpp")
#add_executable(SourceMnist "SourceMnist.cpp")
#add_executable(onnx2trt "onnx2trt.cpp")
add_executable(onnx2trt_gcn "onnx2trt_gcn.cpp")
#add_executable(sample "sample.cpp")
#add_executable(temp "sample.cpp")

#########################################
# test
#add_executable(cublasTest "cublasTest.cpp")
#target_link_libraries(cublasTest cublas)
