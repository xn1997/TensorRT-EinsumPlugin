add_subdirectory(kernels)
AUX_SOURCE_DIRECTORY(./ DIR_LIB_SRCS)
#SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
add_library(einsum_common7_lib ${DIR_LIB_SRCS})
