#include <iostream>
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvUtils.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>
#include "image.hpp"

using namespace std;
using namespace nvinfer1;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) throw() override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

void readPGMFile(const std::string& fileName, uint8_t* buffer, int inH, int inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const{
        if (obj)
        {
            obj->destroy();
        }
    }
};

struct CudaDeleter
{
    void operator()(void* obj){
        if (obj)
        {
            cudaFree(obj);
        }
    }
};

const char* onnxModelFile = "../data/mnist.onnx";
const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

int main()
{
    std::unique_ptr<IBuilder, InferDeleter> builder(createInferBuilder(gLogger));
    builder->setMaxBatchSize(1);

    std::unique_ptr<INetworkDefinition, InferDeleter> network(builder->createNetworkV2(explicitBatch));
    std::unique_ptr<nvonnxparser::IParser, InferDeleter> parser(nvonnxparser::createParser(*network, gLogger));
    parser->parseFromFile(onnxModelFile, static_cast<int>(ILogger::Severity::kWARNING));

    std::unique_ptr<IBuilderConfig, InferDeleter> config(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(1 << 20);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);

    std::unique_ptr<ICudaEngine, InferDeleter> engine(builder->buildEngineWithConfig(*network, *config));
    std::unique_ptr<IExecutionContext, InferDeleter> context(engine->createExecutionContext());

    // 1. 加载图片
    int max_batch_size = 1;
    const char* INPUT_BLOB_NAME = "input";
    const char* OUTPUT_BLOB_NAME = "output";
    int INPUT_H=28,INPUT_W=28,INPUT_C=1;
    int OUTPUT_SIZE = 10;
    int size_input = max_batch_size * INPUT_C * INPUT_H * INPUT_W * sizeof(float);
    int size_output = max_batch_size * OUTPUT_SIZE * sizeof(float);

    std::string image_path = "../data/9.pgm";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat dst = cv::Mat::zeros(INPUT_H, INPUT_W, CV_32FC1);
    cv::resize(img,dst, dst.size());
    float* fileData=normal(img);
//    uchar* fileData = new uchar[28*28];
//    fileData = img.data;
    //2. 定义输入输出
    vector<float> input_cpu_data, output_cpu_data;
    input_cpu_data.resize(INPUT_H*INPUT_W);
    output_cpu_data.resize(OUTPUT_SIZE);

    for(int i = 0; i < INPUT_H*INPUT_W; ++i){
        input_cpu_data[i] = 1.0 - (fileData[i]/255.0);
        std::cout << (" .:-=+*#%@"[static_cast<uchar>(fileData[i]) / 26]) << (((i + 1) % 28) ? "" : "\n");
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* buffers[2];
    cudaMalloc(&buffers[0], size_input);
    cudaMalloc(&buffers[1], size_output);

    cudaMemcpyAsync(buffers[0], input_cpu_data.data(), size_input, cudaMemcpyHostToDevice, stream);

    //  bool is_success = context->executeV2(buffers);
    bool is_success = context->enqueue(1,buffers,stream,nullptr);
    //  bool is_success = context->enqueueV2(buffers,stream,nullptr);
    //  bool is_success = context->execute(1,buffers);
    if(is_success)
        std::cout << "Forward success !" << std::endl;
    else
        std::cout << "Forward Error !" << std::endl;

    cudaMemcpyAsync(output_cpu_data.data(), buffers[1], size_output, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // softmax
    float sum{0.0f};
    for (int i = 0; i < 10; i++){
        output_cpu_data[i] = exp(output_cpu_data[i]);
        sum += output_cpu_data[i];
    }

    // output
    for(int i = 0; i < 10; ++i){
        output_cpu_data[i] /= sum;
        std::cout <<  i << ": " << std::string(floor(output_cpu_data[i] * 10 + 0.5f), '*') << "\n";
    }

    return 1;
}

