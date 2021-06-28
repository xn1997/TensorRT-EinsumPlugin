#include "TrtInfer.h"
void allocate_buffers(std::unique_ptr<ICudaEngine, InferDeleter> &engine,int max_batch_size,
               std::vector<int> &inputIndex,CPU_data &input_cpu_data, CPU_data &output_cpu_data,void** buffers)
{
    // 3.2 分配输入、输出内存（cpu+gpu）
    int NbBindings = engine->getNbBindings();  // number of input+output
//    void* buffers[NbBindings];  // initialize buffers(for gpu data)

    for (int i = 0; i < NbBindings; i++)
    {
        auto dims = engine->getBindingDimensions(i);
        size_t vol = static_cast<size_t>(max_batch_size);
        DataType type = engine->getBindingDataType(i);
        vol *= samplesCommon::volume(dims);
        size_t size_binding = vol * samplesCommon::getElementSize(type);

        cudaMalloc(&buffers[i], size_binding);  // allocate gpu memery
        std::vector<float> temp_data(vol);
        bool is_input = engine->bindingIsInput(i);
        if(is_input){ // 分配
            inputIndex.push_back(i);
            input_cpu_data.push_back(temp_data);  // 创建cpu输入
            input_cpu_data.size.push_back(size_binding);  // 记录输入占用字节数
        }
        else {
            output_cpu_data.push_back(temp_data);
            output_cpu_data.size.push_back(size_binding);
        }
    }
    return;
}
void trt_infer(cudaStream_t &stream,std::unique_ptr<IExecutionContext, InferDeleter> &context,int max_batch_size,
               std::vector<int> &inputIndex,CPU_data &input_cpu_data, CPU_data &output_cpu_data,void** buffers)
{
    auto start_time = std::chrono::system_clock::now();
    for(int i=0; i<inputIndex.size();++i){
        cudaMemcpyAsync(buffers[inputIndex[i]],input_cpu_data[i].data(),input_cpu_data.size[i],cudaMemcpyHostToDevice,stream);
    }
    //  context->enqueue(max_batch_size,buffers,stream,nullptr);
    context->execute(max_batch_size,buffers);
    for(int i=0; i<output_cpu_data.data()->size();i++){
        cudaMemcpyAsync(output_cpu_data[i].data(), buffers[i+inputIndex.size()], output_cpu_data.size[i], cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    auto end_time = std::chrono::system_clock::now();
    std::cout << "infer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;
}
