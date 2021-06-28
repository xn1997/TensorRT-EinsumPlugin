/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "Einsum.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

using namespace nvinfer1;
using nvinfer1::plugin::Einsum;
using nvinfer1::plugin::EinsumCreator;


PluginFieldCollection EinsumCreator::mFC{};
std::vector<PluginField> EinsumCreator::mPluginAttributes;
//REGISTER_TENSORRT_PLUGIN(EinsumCreator);
using namespace nvinfer1::plugin;
// int main(int argc, char** argv)
// {
//     return 0;
// }

namespace
{
constexpr const char* INSTANCE_PLUGIN_VERSION{"1"};
constexpr const char* INSTANCE_PLUGIN_NAME{"Einsum"}; //! 此时对应的plugin就叫CustomEinsum_TRT
} // namespace
#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

#define CHECK_CUDNN(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        cudnnStatus_t status = call;                                                                                   \
        if (status != CUDNN_STATUS_SUCCESS)                                                                            \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

inline bool is_CHW(nvinfer1::Dims const& dims)
{
    return true;
//    (dims.nbDims == 3 && dims.type[0] == nvinfer1::DimensionType::kCHANNEL
//        && dims.type[1] == nvinfer1::DimensionType::kSPATIAL && dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}


/**
 * @brief Einsum::~Einsum 释放op占用的显存
 */
Einsum::~Einsum(){
//    std::cout<< "析构plugin类\t IN ~Plugin" << std::endl;
    terminate();
}
Einsum::Einsum(std::string equation)
    :equation(equation)
{
    std::cout << "构造Plugin\t IN 第一个构造函数，用于parse阶段" << std::endl;
}

Einsum::Einsum(std::string equation, int N, int K, int C, int T, int V, int W)
    :equation(equation),N(N),K(K),C(C),T(T),V(V),W(W)
{
//    std::cout << "构造Plugin\t IN 第二个构造函数，用于clone" << std::endl;
}
//! 反序列化时读入数据
Einsum::Einsum(void const* serialData, size_t serialLength)
{
//    std::cout << "构造Plugin\t IN 第三个构造函数，用于反序列化使用" << std::endl;
    const char *d = reinterpret_cast<const char*>(serialData), *a = d;
    equation = read<std::string>(d);
    N = read<int>(d);
    K = read<int>(d);
    C = read<int>(d);
    T = read<int>(d);
    V = read<int>(d);
    W = read<int>(d);
}


// Einsum returns one output.
int Einsum::getNbOutputs() const throw()
{
    std::cout << "获得输出数量\t IN getNbOutpus" << std::endl;
    return 1;
}

DimsExprs Einsum::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) throw()
{
    std::cout << "计算输出维度\t IN Plugin::getOutputDimensions" << std::endl;
    //! 这里是直接将输入的batch返回
    nvinfer1::DimsExprs output;
    if(equation == "nctkv,kvw->nctw"){
        output.nbDims = 4; //! DimsExprs的d是固定的8维数组，所以必须使用nbDims标记几维
        output.d[0] = inputs[0].d[0];
        output.d[1] = inputs[0].d[1];
        output.d[2] = inputs[0].d[2];
        output.d[3] = inputs[1].d[2];
    }
    return output;
}

int Einsum::initialize() throw()
{
    std::cout << "初始化plugin类\t IN initialize" << std::endl;
    return 0;
}

void Einsum::terminate() throw()
{
}

size_t Einsum::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const throw()
{   //! 这里的显存占用大小是自己估计的
    //! 计算这个op在前向过程中你认为需要的中间显存数量（自己设置）
    std::cout << "获取工作空间大小\t IN getWorkspaceSize" << std::endl;
    size_t need_num = 0;
    size_t res = need_num * sizeof(float); //! 计算这些数量的参数所要占据的内存空间
    return res;
}


int Einsum::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,const nvinfer1::PluginTensorDesc* outputDesc,
                            const void* const* inputs, void* const* outputs,
                            void* workspace,
                            cudaStream_t stream) throw()
{
    printf("error code enter plugin::enqueue %d\n", (int)cudaGetLastError());
    std::cout << "开始推理\t IN enqueue" << endl;
    const float* x = reinterpret_cast<const float*>(inputs[0]); //! 这里inputs[0]不等于inputs[1]-45
    const float* A = reinterpret_cast<const float*>(inputs[1]);

    float* x1 = (float*)malloc(sizeof(float)*N*C*T*K*V);
    float* A1 = (float*)malloc(sizeof(float)*K*V*W);
    cudaMemcpy(x1,x,sizeof(float)*N*C*T*K*V,cudaMemcpyDeviceToHost);
    cudaMemcpy(A1,A,sizeof(float)*K*V*W,cudaMemcpyDeviceToHost);
    float A_sum = 0,x_sum=0;
    std::cout << std::endl << "邻接矩阵A";
    for(int i=0; i<K*V*W; i++){
//        printf("%10.3f",A1[i]);
        A_sum += A1[i];
    }
    std::cout << std::endl << "输入X";
    for(int i=0; i<N*C*T*K*V; i++){
        printf("%10.3f",x1[i]);
        x_sum += x1[i];
    }
    std::cout << std::endl;
    printf("x_sum: %10.3f\tA_sum: %10.3f\n", x_sum,A_sum);
    // 打印输入
    printf("x输入维度为：\t");
    for(int i = 0; i < inputDesc[0].dims.nbDims; ++i){
        std::cout << inputDesc[0].dims.d[i] << ' ';
    }
    printf("A邻接矩阵维度为：\t");
    for(int i = 0; i < inputDesc[1].dims.nbDims; ++i){
        std::cout << inputDesc[1].dims.d[i] << ' ';
    }
    std::cout << std::endl;
//    float* output0 = reinterpret_cast<float*>(outputs[0]);
    cublasHandle_t mCublas; //! 创建cublas句柄
    cublasCreate(&mCublas); //! 这两行是固定写法，不用在意
    float onef{ 1.0f }, zerof{ 0.0f };
    cublasSetStream(mCublas, stream);
    if(equation == "nctkv,kvw->nctw"){
        //! nct,kv * kv,w --> nctw
        //! 矩阵相乘X*A=(AT*XT)T
        cublasSgemm(mCublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    W, N*C*T, K*V,
                    &onef,
                    reinterpret_cast<const float*>(inputs[1]), W,
                    reinterpret_cast<const float*>(inputs[0]), K*V,
                    &zerof,
                    reinterpret_cast<float*>(outputs[0]), W
                    );
    }

    cublasDestroy(mCublas);
    printf("error code leave plugin::enqueue %d\n", (int)cudaGetLastError());
    return 0;
}

size_t Einsum::getSerializationSize() const throw()
{
    std::cout << "获取序列化数据大小\t IN getSerializationSize" << std::endl;
//    return (serialized_size(equation) +
//            serialized_size(N) * 6
//            );
    return sizeof(equation) + sizeof(N) * 6; //! 不能使用上面那句话，会报错，暂不清楚原因
}

//! 将中间变量存入，与deserialize相对应
void Einsum::serialize(void* buffer) const throw()
{
    std::cout << "序列化数据\t IN serialize" << std::endl;
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, equation);
    write(d, N);
    write(d, K);
    write(d, C);
    write(d, T);
    write(d, V);
    write(d, W);
}

//! 检测数据类型和插件格式是否满足要求（自己设计）
bool Einsum::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) throw()
{
    std::cout << "判断输入输出是否符合要求\t IN supportsFormatCombination" << std::endl;
//    ASSERT(inOut && pos < (nbInputs + nbOutputs));
//    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF)
//        && inOut[pos].format == nvinfer1::PluginFormat::kNCHW && inOut[pos].type == inOut[0].type);
    return true;
}

const char* Einsum::getPluginType() const throw()
{
    std::cout << "获取plugin名字\t IN getPluginType" << std::endl;
    return INSTANCE_PLUGIN_NAME;
}

const char* Einsum::getPluginVersion() const throw()
{
    std::cout << "获取plugin版本\t IN getPluginVersion" << std::endl;
    return INSTANCE_PLUGIN_VERSION;
}

void Einsum::destroy() throw()
{
    std::cout << "删除插件类\t IN destroy" << std::endl;
    delete this;
}

IPluginV2DynamicExt* Einsum::clone() const throw()
{
    auto* plugin = new Einsum(equation, N, K, C, T, V, W);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

// Set plugin namespace
void Einsum::setPluginNamespace(const char* pluginNamespace) throw()
{
    mPluginNamespace = pluginNamespace;
}

const char* Einsum::getPluginNamespace() const throw()
{
    return mPluginNamespace.c_str();
}

//! 返回输出数据的类型，和输入数据类型一致
nvinfer1::DataType Einsum::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const throw()
{
    std::cout << "返回输出数据类型\t IN getOutputDataType" << endl;
//    ASSERT(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

void Einsum::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                         const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) throw()
{
    std::cout << "计算中间变量信息\t IN Plugin::configurePlugin" << std::endl;
    N = in[0].desc.dims.d[0];
    C = in[0].desc.dims.d[1];
    T = in[0].desc.dims.d[2];
    K = in[0].desc.dims.d[3];
    V = in[0].desc.dims.d[4];
    W = in[1].desc.dims.d[2];
    cout << "N==>" << N << "\tC==>" << C << "\tT==>" << T << "\tK==>" << K << "\tV==>" << V << "\tW==>" << W << std::endl;
    cout << "K==>" << in[1].desc.dims.d[0] << "\tV==>" << in[1].desc.dims.d[1] << "\tW==>" << in[1].desc.dims.d[2] << std::endl;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Einsum::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) throw()
{
}

// Detach the plugin object from its execution context.
void Einsum::detachFromContext() throw()
{
}

// EinsumCreator methods
//! 对应createPlugin中的操作
//! 初始化plugin field meta data（插件领域的元数据？）
//!     暂且理解为该插件所用到的数据，以关键字+数据的形式存储（即PluginField)
//! PluginField: 以变量名+值的形式存储，该plugin的数据
EinsumCreator::EinsumCreator()
{
    std::cout << "初始化 plugin Creator 类\t IN PluginCreator" << std::endl;
    //! 个人理解---
    //! 使用"equation"的原因是，ONNX模型中Einsum对应的ATTRIBUTES就是equation，可以通过netron查看该结点的信息得到该结论
    //! ONNX该plugin有几个ATTRIBUTES，就在这里添加几个，名字一定要一样，而且数据类型要对应
    mPluginAttributes.emplace_back(PluginField("equation", nullptr, PluginFieldType::kCHAR, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EinsumCreator::getPluginName() const throw()
{
    return INSTANCE_PLUGIN_NAME;
}

const char* EinsumCreator::getPluginVersion() const throw()
{
    return INSTANCE_PLUGIN_VERSION;
}

const PluginFieldCollection* EinsumCreator::getFieldNames() throw()
{
    std::cout << "获取插件信息\t IN getFieldNames" << std::endl;
    return &mFC;
}

//!
//! 可以认为这里是整个解析的开始，在这个函数中创建类plugin类，为其传递类equation参数-->进而才能进行plugin类内的各种操作
//! \brief EinsumCreator::createPlugin 从parse中解析到的数据(PluginFieldCollection类型的数据)，生成插件类(Einsum)
//! \param name
//! \param fc
//! \return
//!
IPluginV2DynamicExt* EinsumCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) throw()
{
    std::cout << "解析ONNX数据 并构建plugin类\t IN PluginCreator::createPlugin" << std::endl;
    const char* mequation;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attrName = fields[i].name; //! 读取数据名
        if (!strcmp(attrName, "equation")) {
            // assert(fields[i].type == PluginFieldType::kCHAR);
            // strcpy(mequation,*(static_cast<const std::string*>(fields[i].data)));
            mequation = static_cast<const char*>(fields[i].data); //! 读取数据
        }
    }
    std::cout << "equation is " << mequation << std::endl;
    //! 创建plugin
    Einsum* obj = new Einsum(mequation);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2DynamicExt* EinsumCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) throw()
{
    Einsum* obj = new Einsum(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
