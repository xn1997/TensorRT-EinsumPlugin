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
#ifndef EINSUM_H
#define EINSUM_H
#include "plugin.h"
#include "serialize.hpp"
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>

typedef unsigned short half_type;

namespace nvinfer1
{
namespace plugin
{
//! 加上final，后续不可以对该类(Einsum)进行继承
class Einsum final : public nvinfer1::IPluginV2DynamicExt
{

public:
    /** 03
     * @brief Einsum 用在parse阶段，用于创建该插件(op)时调用的构造函数，需要传递权值及参数
     */
    Einsum(std::string equation);
    /** 04
     * @brief Einsum 用于clone阶段，复制这个plugin时用到该构造函数
     */
    Einsum(std::string equation, int N, int K, int C, int T, int V, int W);
    /**
     * @brief Einsum 用于反序列化(deserialize)阶段，将序列化好的权重和参数传入该plugin并创建该op
     * @param serialData
     * @param serialLength
     */
    Einsum(void const* serialData, size_t serialLength);

    Einsum() = delete;  //! 默认构造函数必须删掉

    ~Einsum() override; //! 调用terminate释放显存，完成析构


    //!
    //! \brief clone 将这个plugin对象克隆一份给TensorRT的builder、network或者engine
    //!     会调用上述的第二个构造函数，完成克隆
    //!     主要用于传递不变的权重和参数，将plugin复制多份，从而被不同的engine、builder或者network使用
    //! \return
    //!
    nvinfer1::IPluginV2DynamicExt* clone() const throw() override;


    /**
     * @brief getNbOutputs 该op返回多少个Tensor(即几个output），一般直接返回1(一个输出)
     * @return
     */
    int getNbOutputs() const throw() override;

    /**
     * @brief getOutputDataType 返回输出的数据类型（包括float, half[float16], int 8, int32, bool)
     *      一般直接返回输入数据的数据类型，即输入输出数据类型一致
     * @param index
     * @param inputTypes
     * @param nbInputs
     * @return
     */
    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const throw() override;


    //!
    //! \brief getOutputDimensions 获得batch的维度
    //!     TensorRT在支持Dynamic-shape时，batch这一维度必须是explicit，也就是说，TensorRT处理的维度从以往的三维[3,-1,-1]变成了[1,3,-1,-1]。
    //! 该函数就是返回batch的维度，上面的例子就是返回1。
    //!     我们在这个成员函数中要做的就是根据输入维度推理出该op的输出维度（一般直接将输入维度作为输出的维度即可）
    //!     注意：虽然输出维度是由输入维度决定的，但这个输出维度其实是内定的（也就是在计算之前已经算出来了）。
    //! 如果该op的输出维度需要根据实际运行计算得到，是不可以的。
    //! \param outputIndex
    //! \param inputs
    //! \param nbInputs
    //! \param exprBuilder
    //! \return
    //!
    // DynamicExt plugins returns DimsExprs class instead of Dims
    DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) throw() override;


    /**
     * @brief initialize 初始化函数，在engine创建时调用(即op准备开始run之前执行)
     *      主要初始化一些提前开辟空间的参数，一般都是cuda操作需要的参数（比如conv需要提前开辟weight和bias的显存），
     * 算子需要这些参数，就必须提前在这里开辟显存。
     *      注意：如果该op算子需要开辟比较大的显存空间，尽量不要自己去申请显存空间，可以使用TensorRT官方接口传过来的workspace指针来获取显存空间。
     * 因为如果该op被一个网络调用很多次，而这个op需要开辟很多显存空间，那么TensorRT在构建network时就会根据这个插件被调用的次数开辟很多显存，
     * 导致显存溢出。(--self：而使用workspace指针，就保证每个op都使用的同一块地址，不会导致显存溢出。所有op都在同一块空间运算，运算完一个op，
     * 将下一个op的参数放入该空间，执行下一个op，各个op的数据无需保留，所以运行下一个op直接清除上一个op的数据即可)
     * @return
     */
    int initialize() throw() override;


    /**
     * @brief terminate 释放该op开辟的一些显存空间——用于析构函数，在engine被摧毁时调用
     */
    void terminate() throw() override;


    /**
     * @brief getWorkspaceSize 返回该op需要的(临时显存大小)中间显存变量的实际数据大小(byte size)——一般都是使用该官方函数获取，比较规范
     *      在这里确定这个op需要多大的显存空间去运行，这样在实际运行时就可以直接使用TensorRT开辟好的空间而不是自己去申请显存空间，避免上述显存溢出的问题。
     * @param inputs
     * @param nbInputs
     * @param outputs
     * @param nbOutputs
     * @return
     */
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const throw() override;


    /**
     * @brief enqueue op实际执行时运行的函数
     *      将自己实现的计算过程放在该函数内，一般为cuda实现操作（C++实现的op操作也可以放进来，不过因为时cpu执行，速度比较慢）
     *      根据输入inputs计算输出outputs，传给相应的指针即可。
     *      注意：如果op需要在显存中暂存一些中间变量，可以通过传进来的指针参数workspace获取
     *      默认写的.cu是FP32精度，当TensorRT在FP16运行模式下，运行到不支持FP16的插件op时，会自动切换到FP32模式，等op运行完再切换回FP16，
     * 因此，这样频繁的数据转换也会造成时间的消耗增加
     * @param inputDesc
     * @param outputDesc
     * @param inputs
     * @param outputs
     * @param workspace 分配的工作空间的地址，通过该地址，来暂存op计算需要的中间变量，避免重复单独开辟空间
     * @param stream
     * @return
     */
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs,
                void* workspace,
                cudaStream_t stream) throw() override;


    //!
    //! \brief setPluginNamespace 为这个插件设置namespace名字，默认是""。（一般不设置）
    //!     注意：同一个namepace下的plugin如果名字相同会发生冲突。
    //!     就是修改 mPluginNamespace
    //! \param pluginNamespace
    //!
    void setPluginNamespace(const char* pluginNamespace) throw() override;
    const char* getPluginNamespace() const throw() override;    //! 获取该plugin的namespace名字


    //!
    //! \brief configurePlugin 判断输入和输出的数据类型是否正确。也可以通过这个配置信息告诉TensorRT去选择合适的算法来调优这个模型——貌似也负责计算相关的中间变量
    //!     我们一般写的plugin执行代码都是固定的，不要调优，所以这个主要是针对官方的op
    //! \param in
    //! \param nbInputs
    //! \param out
    //! \param nbOutputs
    //!
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) throw() override;


    //!
    //! \brief getSerializationSize 返回序列化(serialize)该op时需要写多少字节到buffer中
    //!     一般为权重+参数的总的字节数
    //! \return
    //!
    size_t getSerializationSize() const throw() override;


    //!
    //! \brief serialize 根据序列化大小getSerializationSize()，把需要用到的数据按照顺序序列化到buffer中
    //!     就是指权重+参数+额外的内存空间(应该是中间变量吧)序列化到buffer中
    //! \param buffer
    //!
    void serialize(void* buffer) const throw() override;


    // DynamicExt plugin supportsFormat update.
    //!
    //! \brief supportsFormatCombination TensorRT调用该方法来判断pos索引对应的(输入/输出)是否支持inOut[pos].format和inOut[pos].type指定的格式和数据类型
    //!     知乎有说，但是每读懂
    //! 暂且认为判断输入/输出的格式和数据类型是否满足要求
    //!     数据类型指DataType：float, half[float16], int 8, int32, bool
    //!     格式指TensorFormat：kLINEAR，kNCHW，kNCHW2，等（暂时不懂具体啥区别）
    //! \param pos
    //! \param inOut
    //! \param nbInputs
    //! \param nbOutputs
    //! \return
    //!
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) throw() override;


    //!
    //! \brief attachToContext 如果这个op使用到了一些其他东西，例如cublas handle，可以直接借助TensorRT内部提供的cublas handle
    //!     暂时不清楚如何使用
    //! \param cudnn
    //! \param cublas
    //! \param allocator
    //!
    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) throw() override;

    //!
    //! \brief getPluginType 自己设置该plugin的名字和版本号（比如leakyrelu 1)
    //!     注意：由于PluginNamespace一般默认为"",所以这里的plugin必须不能重复，否则编译报错(????待测试）
    //! \return
    //!
    const char* getPluginType() const throw() override;
    const char* getPluginVersion() const throw() override;


    void destroy() throw() override;

    void detachFromContext() throw() override;

private:
    std::string equation;
    int N,K,C,T,V,W;
    std::string mNamespace;
//    const char* mPluginNamespace;   //! 该plugin的namepace名字，一般不设置，为""即可
    std::string mPluginNamespace;
};

class EinsumCreator : public BaseCreator
{
public:
    //!
    //! \brief EinsumCreator
    //! 01
    EinsumCreator();

    ~EinsumCreator() override = default;


    //!
    //! \brief getPluginName 对应Plugin插件类中的getPluginType，getPluginVersion。 一模一样
    //! \return
    //!
    const char* getPluginName() const throw() override;
    const char* getPluginVersion() const throw() override;


    //!
    //! \brief getFieldNames
    //! \param PluginFieldCollection mFC: 这是成员变量
    //!     主要作用是传递这个op所需要的权重和参数，在engine推理时不会使用，而在parse中使用（比如caffe2trt,onnx2trt），决定了是否可以解析成功。
    //! 当使用parse解析这个op时，这个op的权重和参数会经历Models-->TensorRT engine --> TensorRT runtime的过程。
    //!     具体过程参考知乎链接
    //! \return
    //!
    const PluginFieldCollection* getFieldNames() throw() override;


    //!
    //! \brief createPlugin 通过得到的PluginFieldCollection去创建一个plugin，将op需要的权重和参数取出来，然后调用插件类的第一个构造函数，来创建plugin
    //!
    //! 另一种理解（参考https://github.com/LitLeo/TensorRT_Tutorial/blob/master/blogs/TensorRT%20Plugin%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8F%E7%AE%80%E4%BB%8B-%E4%BB%A5leaky%20relu%E5%B1%82%E4%B8%BA%E4%BE%8B.md）
    //!     根据序列化数据，反序列化为plugin类。
    //! 需要的参数有：
    //!     plugin的名字，该参数非常重要，是反序列化为哪种plugin的唯一凭证
    //!     序列化数据
    //!     序列化后的数据的长度
    //! \param name 该plugin的名字
    //! \param fc   序列化所需的数据
    //! \return
    //! 02
    IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) throw() override;


    //!
    //! \brief deserializePlugin 将op读取到的onnx模型的data数据反序列化到network中。调用插件类的第三个构造函数，来创建plugin
    //! \param name
    //! \param serialData
    //! \param serialLength
    //! \return
    //!
    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) throw() override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_INSTANCE_NORMALIZATION_PLUGIN_H
