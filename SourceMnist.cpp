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

using namespace std;
using namespace nvinfer1;

class Logger : public ILogger
{
void log(Severity severity, const char* msg) override
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

  vector<float> input_cpu_data, output_cpu_data;
  input_cpu_data.resize(28*28);
  output_cpu_data.resize(10);

  // Load pgm image
  std::vector<uint8_t> fileData(28 * 28);
  readPGMFile("../data/9.pgm", fileData.data(), 28, 28);
  for(int i = 0; i < 28*28; ++i){
    input_cpu_data[i] = 1.0 - (fileData[i]/255.0);
    std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % 28) ? "" : "\n");
  }


  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int size_input = sizeof(float)*28*28;
  int size_output = sizeof(float)*10;

  void *input_gpu_data_ptr, *output_gpu_data_ptr;
  std::unique_ptr<void, CudaDeleter> input_gpu_data, output_gpu_data;
  cudaMalloc(&input_gpu_data_ptr, size_input);
  cudaMalloc(&output_gpu_data_ptr, size_output);
  input_gpu_data.reset(input_gpu_data_ptr);
  output_gpu_data.reset(output_gpu_data_ptr);

  void* buffers[2];
  buffers[0] = input_gpu_data.get();
  buffers[1] = output_gpu_data.get();

  cudaMemcpyAsync(buffers[0], input_cpu_data.data(), size_input, cudaMemcpyHostToDevice, stream);

  bool is_success = context->executeV2(buffers);
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

