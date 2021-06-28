cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 11)

# include_directories(/home/xzy/Data2/xzyLinuxInstallPackage/5TensorRT/TensorRT-7.2.1.6/include/)
# file(GLOB_RECURSE trtlib /home/xzy/Data2/xzyLinuxInstallPackage/5TensorRT/TensorRT-7.2.1.6/lib/*.so)
set(trthead /home/xzy/Data2/xzyLinuxInstallPackage/5TensorRT/TensorRT-8.0.0.3/include/)
file(GLOB_RECURSE trtlib /home/xzy/Data2/xzyLinuxInstallPackage/5TensorRT/TensorRT-8.0.0.3/lib/*.so)
message("tensort lib: in " ${trtlib})

find_package(CUDA REQUIRED)
file(GLOB_RECURSE CUDA_LIBRARIES /usr/local/cuda/lib64/*.so) # 我的电脑不知道为什么so文件导入不全，所以这里自行导入所有so，如果你的环境没问题可以不用加这一句
message("cuda include : " ${CUDA_INCLUDE_DIRS})
message("cuda lib : " ${CUDA_LIBRARIES})


# include_directories(./einsum_common7)
# add_subdirectory(einsum_common7 einsum_common7)
# link_libraries(einsum_common7_lib)

include_directories(${trthead} ${CUDA_INCLUDE_DIRS}) # 单独链接cuda，为了einsum_common可以调用cuda库
link_libraries(${trtlib} ${CUDA_LIBRARIES})
add_subdirectory(einsum_common8 einsum_common8)

add_library(einsumlib Einsum.cpp)
set (Einsum_Heads
    ${trthead}
    ${CUDA_INCLUDE_DIRS}
    einsum_common8
)
set (Einsum_Libs
    ${trtlib}
    ${CUDA_LIBRARIES}
    einsum_common8_lib
)
target_include_directories(einsumlib PUBLIC ${Einsum_Heads}) # 使用PUBLIC，使其可以往上传递，其链接的库都可以被使用einsumlib的库使用
target_link_libraries(einsumlib PUBLIC ${Einsum_Libs})
