add_subdirectory(kernels)
aux_source_directory(./ DIR_LIB_SRCS)
add_library(einsum_common8_lib ${DIR_LIB_SRCS})