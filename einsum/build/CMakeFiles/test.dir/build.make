# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/888/bin/cmake

# The command to remove a file.
RM = /snap/cmake/888/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/build

# Include any dependencies generated for this target.
include CMakeFiles/test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test.dir/flags.make

CMakeFiles/test.dir/onnx2trt_gcn.cpp.o: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/onnx2trt_gcn.cpp.o: ../onnx2trt_gcn.cpp
CMakeFiles/test.dir/onnx2trt_gcn.cpp.o: CMakeFiles/test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test.dir/onnx2trt_gcn.cpp.o"
	/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test.dir/onnx2trt_gcn.cpp.o -MF CMakeFiles/test.dir/onnx2trt_gcn.cpp.o.d -o CMakeFiles/test.dir/onnx2trt_gcn.cpp.o -c /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/onnx2trt_gcn.cpp

CMakeFiles/test.dir/onnx2trt_gcn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test.dir/onnx2trt_gcn.cpp.i"
	/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/onnx2trt_gcn.cpp > CMakeFiles/test.dir/onnx2trt_gcn.cpp.i

CMakeFiles/test.dir/onnx2trt_gcn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test.dir/onnx2trt_gcn.cpp.s"
	/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/onnx2trt_gcn.cpp -o CMakeFiles/test.dir/onnx2trt_gcn.cpp.s

CMakeFiles/test.dir/Einsum.cpp.o: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/Einsum.cpp.o: ../Einsum.cpp
CMakeFiles/test.dir/Einsum.cpp.o: CMakeFiles/test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/test.dir/Einsum.cpp.o"
	/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test.dir/Einsum.cpp.o -MF CMakeFiles/test.dir/Einsum.cpp.o.d -o CMakeFiles/test.dir/Einsum.cpp.o -c /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/Einsum.cpp

CMakeFiles/test.dir/Einsum.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test.dir/Einsum.cpp.i"
	/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/Einsum.cpp > CMakeFiles/test.dir/Einsum.cpp.i

CMakeFiles/test.dir/Einsum.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test.dir/Einsum.cpp.s"
	/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/Einsum.cpp -o CMakeFiles/test.dir/Einsum.cpp.s

# Object files for target test
test_OBJECTS = \
"CMakeFiles/test.dir/onnx2trt_gcn.cpp.o" \
"CMakeFiles/test.dir/Einsum.cpp.o"

# External object files for target test
test_EXTERNAL_OBJECTS =

test: CMakeFiles/test.dir/onnx2trt_gcn.cpp.o
test: CMakeFiles/test.dir/Einsum.cpp.o
test: CMakeFiles/test.dir/build.make
test: /usr/local/cuda/lib64/libOpenCL.so
test: /usr/local/cuda/lib64/libaccinj64.so
test: /usr/local/cuda/lib64/libcublas.so
test: /usr/local/cuda/lib64/libcublasLt.so
test: /usr/local/cuda/lib64/libcudart.so
test: /usr/local/cuda/lib64/libcudnn.so
test: /usr/local/cuda/lib64/libcudnn_adv_infer.so
test: /usr/local/cuda/lib64/libcudnn_adv_train.so
test: /usr/local/cuda/lib64/libcudnn_cnn_infer.so
test: /usr/local/cuda/lib64/libcudnn_cnn_train.so
test: /usr/local/cuda/lib64/libcudnn_ops_infer.so
test: /usr/local/cuda/lib64/libcudnn_ops_train.so
test: /usr/local/cuda/lib64/libcufft.so
test: /usr/local/cuda/lib64/libcufftw.so
test: /usr/local/cuda/lib64/libcuinj64.so
test: /usr/local/cuda/lib64/libcupti.so
test: /usr/local/cuda/lib64/libcurand.so
test: /usr/local/cuda/lib64/libcusolver.so
test: /usr/local/cuda/lib64/libcusolverMg.so
test: /usr/local/cuda/lib64/libcusparse.so
test: /usr/local/cuda/lib64/libnppc.so
test: /usr/local/cuda/lib64/libnppial.so
test: /usr/local/cuda/lib64/libnppicc.so
test: /usr/local/cuda/lib64/libnppidei.so
test: /usr/local/cuda/lib64/libnppif.so
test: /usr/local/cuda/lib64/libnppig.so
test: /usr/local/cuda/lib64/libnppim.so
test: /usr/local/cuda/lib64/libnppist.so
test: /usr/local/cuda/lib64/libnppisu.so
test: /usr/local/cuda/lib64/libnppitc.so
test: /usr/local/cuda/lib64/libnpps.so
test: /usr/local/cuda/lib64/libnvToolsExt.so
test: /usr/local/cuda/lib64/libnvblas.so
test: /usr/local/cuda/lib64/libnvjpeg.so
test: /usr/local/cuda/lib64/libnvperf_host.so
test: /usr/local/cuda/lib64/libnvperf_target.so
test: /usr/local/cuda/lib64/libnvrtc-builtins.so
test: /usr/local/cuda/lib64/libnvrtc.so
test: /usr/local/cuda/lib64/stubs/libcublas.so
test: /usr/local/cuda/lib64/stubs/libcublasLt.so
test: /usr/local/cuda/lib64/stubs/libcuda.so
test: /usr/local/cuda/lib64/stubs/libcufft.so
test: /usr/local/cuda/lib64/stubs/libcufftw.so
test: /usr/local/cuda/lib64/stubs/libcurand.so
test: /usr/local/cuda/lib64/stubs/libcusolver.so
test: /usr/local/cuda/lib64/stubs/libcusolverMg.so
test: /usr/local/cuda/lib64/stubs/libcusparse.so
test: /usr/local/cuda/lib64/stubs/libnppc.so
test: /usr/local/cuda/lib64/stubs/libnppial.so
test: /usr/local/cuda/lib64/stubs/libnppicc.so
test: /usr/local/cuda/lib64/stubs/libnppidei.so
test: /usr/local/cuda/lib64/stubs/libnppif.so
test: /usr/local/cuda/lib64/stubs/libnppig.so
test: /usr/local/cuda/lib64/stubs/libnppim.so
test: /usr/local/cuda/lib64/stubs/libnppist.so
test: /usr/local/cuda/lib64/stubs/libnppisu.so
test: /usr/local/cuda/lib64/stubs/libnppitc.so
test: /usr/local/cuda/lib64/stubs/libnpps.so
test: /usr/local/cuda/lib64/stubs/libnvidia-ml.so
test: /usr/local/cuda/lib64/stubs/libnvjpeg.so
test: /usr/local/cuda/lib64/stubs/libnvrtc.so
test: einsum_common8/libeinsum_common8_lib.a
test: /usr/local/cuda/lib64/libOpenCL.so
test: /usr/local/cuda/lib64/libaccinj64.so
test: /usr/local/cuda/lib64/libcublas.so
test: /usr/local/cuda/lib64/libcublasLt.so
test: /usr/local/cuda/lib64/libcudart.so
test: /usr/local/cuda/lib64/libcudnn.so
test: /usr/local/cuda/lib64/libcudnn_adv_infer.so
test: /usr/local/cuda/lib64/libcudnn_adv_train.so
test: /usr/local/cuda/lib64/libcudnn_cnn_infer.so
test: /usr/local/cuda/lib64/libcudnn_cnn_train.so
test: /usr/local/cuda/lib64/libcudnn_ops_infer.so
test: /usr/local/cuda/lib64/libcudnn_ops_train.so
test: /usr/local/cuda/lib64/libcufft.so
test: /usr/local/cuda/lib64/libcufftw.so
test: /usr/local/cuda/lib64/libcuinj64.so
test: /usr/local/cuda/lib64/libcupti.so
test: /usr/local/cuda/lib64/libcurand.so
test: /usr/local/cuda/lib64/libcusolver.so
test: /usr/local/cuda/lib64/libcusolverMg.so
test: /usr/local/cuda/lib64/libcusparse.so
test: /usr/local/cuda/lib64/libnppc.so
test: /usr/local/cuda/lib64/libnppial.so
test: /usr/local/cuda/lib64/libnppicc.so
test: /usr/local/cuda/lib64/libnppidei.so
test: /usr/local/cuda/lib64/libnppif.so
test: /usr/local/cuda/lib64/libnppig.so
test: /usr/local/cuda/lib64/libnppim.so
test: /usr/local/cuda/lib64/libnppist.so
test: /usr/local/cuda/lib64/libnppisu.so
test: /usr/local/cuda/lib64/libnppitc.so
test: /usr/local/cuda/lib64/libnpps.so
test: /usr/local/cuda/lib64/libnvToolsExt.so
test: /usr/local/cuda/lib64/libnvblas.so
test: /usr/local/cuda/lib64/libnvjpeg.so
test: /usr/local/cuda/lib64/libnvperf_host.so
test: /usr/local/cuda/lib64/libnvperf_target.so
test: /usr/local/cuda/lib64/libnvrtc-builtins.so
test: /usr/local/cuda/lib64/libnvrtc.so
test: /usr/local/cuda/lib64/stubs/libcublas.so
test: /usr/local/cuda/lib64/stubs/libcublasLt.so
test: /usr/local/cuda/lib64/stubs/libcuda.so
test: /usr/local/cuda/lib64/stubs/libcufft.so
test: /usr/local/cuda/lib64/stubs/libcufftw.so
test: /usr/local/cuda/lib64/stubs/libcurand.so
test: /usr/local/cuda/lib64/stubs/libcusolver.so
test: /usr/local/cuda/lib64/stubs/libcusolverMg.so
test: /usr/local/cuda/lib64/stubs/libcusparse.so
test: /usr/local/cuda/lib64/stubs/libnppc.so
test: /usr/local/cuda/lib64/stubs/libnppial.so
test: /usr/local/cuda/lib64/stubs/libnppicc.so
test: /usr/local/cuda/lib64/stubs/libnppidei.so
test: /usr/local/cuda/lib64/stubs/libnppif.so
test: /usr/local/cuda/lib64/stubs/libnppig.so
test: /usr/local/cuda/lib64/stubs/libnppim.so
test: /usr/local/cuda/lib64/stubs/libnppist.so
test: /usr/local/cuda/lib64/stubs/libnppisu.so
test: /usr/local/cuda/lib64/stubs/libnppitc.so
test: /usr/local/cuda/lib64/stubs/libnpps.so
test: /usr/local/cuda/lib64/stubs/libnvidia-ml.so
test: /usr/local/cuda/lib64/stubs/libnvjpeg.so
test: /usr/local/cuda/lib64/stubs/libnvrtc.so
test: CMakeFiles/test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test.dir/build: test
.PHONY : CMakeFiles/test.dir/build

CMakeFiles/test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test.dir/clean

CMakeFiles/test.dir/depend:
	cd /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/build /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/build /home/xzy/G/DeepLearning/Gitee/TensorRT/CPP/TensorRT/einsum/build/CMakeFiles/test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test.dir/depend

