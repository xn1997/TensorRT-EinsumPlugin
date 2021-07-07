cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 11)

include_directories(/home/xzy/Data2/xzyLinuxInstallPackage/5TensorRT/TensorRT-7.2.1.6/include/)
file(GLOB_RECURSE trtlib /home/xzy/Data2/xzyLinuxInstallPackage/5TensorRT/TensorRT-7.2.1.6/lib/*.so)
# set(trthead /home/xzy/Data2/xzyLinuxInstallPackage/5TensorRT/TensorRT-8.0.0.3/include/)
# file(GLOB_RECURSE trtlib /home/xzy/Data2/xzyLinuxInstallPackage/5TensorRT/TensorRT-8.0.0.3/lib/*.so)
message("tensort lib: in " ${trtlib})

find_package(CUDA REQUIRED)
file(GLOB_RECURSE CUDA_LIBRARIES /usr/local/cuda/lib64/*.so) # 我的电脑不知道为什么so文件导入不全，所以这里自行导入所有so，如果你的环境没问题可以不用加这一句
message("cuda include : " ${CUDA_INCLUDE_DIRS})
message("cuda lib : " ${CUDA_LIBRARIES})

include_directories(${trthead} ${CUDA_INCLUDE_DIRS}) # 单独链接cuda，为了einsum_common可以调用cuda库
link_libraries(${trtlib} ${CUDA_LIBRARIES})

set(einsum_common_dir einsum_common7) # 设置einsum_common的版本，即tensorrt的版本，目前只支持tensorrt的8和7版本
add_subdirectory(${einsum_common_dir} einsum_common)

add_library(einsumlib Einsum.cpp)
set (Einsum_Heads
    ${trthead}
    ${CUDA_INCLUDE_DIRS}
    ${einsum_common_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set (Einsum_Libs
    ${trtlib}
    ${CUDA_LIBRARIES}
    einsum_common_lib
)
target_include_directories(einsumlib PUBLIC ${Einsum_Heads}) # 使用PUBLIC，使其可以往上传递，其链接的库都可以被使用einsumlib的库使用
target_link_libraries(einsumlib PUBLIC ${Einsum_Libs})
