//#ifndef FUNCTION_H
//#define FUNCTION_H
#pragma once
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

#include <assert.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <unistd.h>
#include "buffers.h"
#include "image.hpp"

class CPU_data :public std::vector<std::vector<float>>
{
public:
    CPU_data() {}
    std::vector<size_t> size;  // 记录每个数据的内存占用字节数 *sizeof(float)
};

using namespace nvinfer1;
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) throw() override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
        std::cout << msg << std::endl;
    }
};

extern Logger gLogger;
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
//constexpr long double operator"" _GiB(long double val)
//{
//    return val * (1 << 30);
//}
//constexpr long double operator"" _MiB(long double val)
//{
//    return val * (1 << 20);
//}
//constexpr long double operator"" _KiB(long double val)
//{
//    return val * (1 << 10);
//}
std::vector<std::vector<float>> reshape_1to2D(std::vector<int> shape,std::vector<float> data);
void printfVector2D(std::vector<std::vector<float>> arrays,int max_lengths);

ICudaEngine* loadEngine(const std::string& engine);

//#endif // FUNCTION_H
