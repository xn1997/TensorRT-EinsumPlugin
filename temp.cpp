#include "function.h"
#include <algorithm>

using namespace std;
using namespace nvinfer1;

std::string image_path = "../data/tabby_tiger_cat.jpg";
const char* onnxModelFile = "/home/xzy/G/GraduateStudentProject/HitachiEscalator/SVM/GCN2TRT/gcn.onnx";
std::string engine_save_path = "../data/gcn.engine";

const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

// 0. 参数
int max_batch_size = 1;
int main()
{
    std::unique_ptr<IBuilder, InferDeleter> builder(createInferBuilder(gLogger));
    builder->setMaxBatchSize(1);

    std::unique_ptr<INetworkDefinition, InferDeleter> network(builder->createNetworkV2(explicitBatch));
    std::unique_ptr<nvonnxparser::IParser, InferDeleter> parser(nvonnxparser::createParser(*network, gLogger));
    parser->parseFromFile(onnxModelFile, static_cast<int>(ILogger::Severity::kWARNING));

    std::unique_ptr<IBuilderConfig, InferDeleter> config(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(1_GiB);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
//    config->set
//    config->setFlag(BuilderFlag::kTF32);
//    builder->setHalf2Mode(false);

    std::unique_ptr<ICudaEngine, InferDeleter> engine(builder->buildEngineWithConfig(*network, *config));
    // 4. 保存engine（文件名后缀可任意定）
    IHostMemory* modelStream = engine->serialize();
    std::ofstream p(engine_save_path,std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    printf("modelStream->size():%d\n", modelStream->size());
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
//    return 0;
    // 5. 生成context
    std::unique_ptr<IExecutionContext, InferDeleter> context(engine->createExecutionContext());

    // 3. 推理
    // 3.1 构建stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 3.2 分配输入、输出内存（cpu+gpu）
    std::vector<int> inputIndex;  // 输入索引
    CPU_data input_cpu_data, output_cpu_data;
    int NbBindings = engine->getNbBindings();  // number of input+output
    void* buffers[NbBindings];  // initialize buffers(for gpu data)

    for (int i = 0; i < NbBindings; i++)
    {
        auto dims = engine->getBindingDimensions(i);
        size_t vol = static_cast<size_t>(max_batch_size);
        DataType type = engine->getBindingDataType(i);
        vol *= samplesCommon::volume(dims);
        size_t size_binding = vol * samplesCommon::getElementSize(type);

        cudaMalloc(&buffers[i], size_binding);  // allocate gpu memery
        vector<float> temp_data(vol);
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
    // 3.3 加载输入
    int N=1,C=3,T=1,K=1,V=15;
    for(int i = 0; i < N*C*T*K*V; ++i){
        input_cpu_data[0][i] = 1.0;
    }
//    float* float = (float*)malloc(N*C*T*K*V*sizeof(float));
//    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
//    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
//    cv::Mat dst = cv::Mat::zeros(INPUT_H, INPUT_W, CV_32FC3);
//    cv::resize(img,dst, dst.size());
//    float* fileData=normal(dst);
//    for(int i = 0; i < INPUT_H*INPUT_W*INPUT_C; ++i){
//        input_cpu_data[0][i] = fileData[i];
//    }
//    free(fileData);  // 释放图片
    auto start_time = std::chrono::system_clock::now();
    // 3.2 输入从cpu拷贝至gpu
    for(int i=0; i<inputIndex.size();++i){
        cudaMemcpyAsync(buffers[inputIndex[i]],input_cpu_data[i].data(),input_cpu_data.size[i],cudaMemcpyHostToDevice,stream);
    }
    // 3.3 infer 即在gpu执行kernel
    bool is_success = context->enqueue(max_batch_size,buffers,stream,nullptr);
    if(is_success)
        std::cout << "Forward success !" << std::endl;
    else
        std::cout << "Forward Error !" << std::endl;
    // 3.4 输出从gpu拷贝至cpu
    for(int i=0; i<NbBindings-inputIndex.size();i++){
        cudaMemcpyAsync(output_cpu_data[i].data(), buffers[i+inputIndex.size()], output_cpu_data.size[i], cudaMemcpyDeviceToHost, stream);
    }
    // 3.5 同步stream，并销毁
    cudaStreamSynchronize(stream);
    auto end_time = std::chrono::system_clock::now();
    std::cout << "infer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;
    cudaStreamDestroy(stream);

    for(std::vector<float>::iterator it=output_cpu_data[0].begin(); it!=output_cpu_data[0].end();it++){
//        std::cout << *it << std::endl;
    }
    std::cout << output_cpu_data[0].size() <<"\t " << \
                                             std::accumulate(output_cpu_data[0].begin(), output_cpu_data[0].end(),0) << std::endl;
    if(std::accumulate(output_cpu_data[0].begin(), output_cpu_data[0].end(),0) != 675){
//        return 0;
    }
    return 1;
}
