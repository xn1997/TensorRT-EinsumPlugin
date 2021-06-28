#pragma once
#include "function.h"
void allocate_buffers(std::unique_ptr<ICudaEngine, InferDeleter> &engine,int max_batch_size,
               std::vector<int> &inputIndex,CPU_data &input_cpu_data, CPU_data &output_cpu_data,void** buffers);
void trt_infer(cudaStream_t &stream,std::unique_ptr<IExecutionContext, InferDeleter> &context,int max_batch_size,
               std::vector<int> &inputIndex,CPU_data &input_cpu_data, CPU_data &output_cpu_data,void** buffers);
