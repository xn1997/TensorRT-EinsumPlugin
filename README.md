## 前言

由于TensorRT并未实现Einsum插件，而在转换GCN的过程中，网络频繁使用该op，因此，不得已手写plugin，顺便也学习了一下plugin的编写及注册方法。

在plugin文件内，对每个成员函数的功能都进行了简单的注释（可能会有很多不清楚的地方，建议自行百度详细了解一下）

强烈推荐这个教程[实现TensorRT自定义插件(plugin)自由](https://zhuanlan.zhihu.com/p/297002406)，按照这个教程肯定可以生成可以用的Plugin，目前网上搜到的都是直接重新编译生成新的`libnvinfer_plugin.so`替换官方原有的该库，然而这种方法在TensorRT-OSS版本不匹配TensorRT时就不能使用了，因此这里采用另一种更加灵活的方法：直接将EinsumPlugin编译到自己的项目工程里即可。

## 环境

> TensorRT8.0

TensorRT8.0相比以前版本更新了很多，因此该版本编写的Einsum不可以用于其他版本（其实就是在7.0版本的基础上，将所有的成员函数后加一个`throw()`就可以了，看程序就懂了）

如果想要使用TensorRT7，直接在`einsum/CMakeLists.txt`内修改TensorRT路径，并切换到`einsum_common7`即可

## 使用流程

参考根目录下的`CMakeLists.txt`将einsum添加到自己的项目之中即可。

==记得一定要在模型解析的源文件内(如这里的onnx2trt_gcn.cpp)，使用`REGISTER_TENSORRT_PLUGIN(EinsumCreator)`来注册Einsum插件，这样解析时才可以找到==

**测试用例使用方法**

1. 运行`generate_onnx.py`生成测试的onnx文件
2. 编译该仓库，然后运行`onnnx2trt_gcn`，如果每报错就说明该einsum插件没有问题

## 编写流程

1. 从TensorRT-OSS的plugin文件内，拷贝一个官方的例程，然后直接在里面的编写就可以了，替换掉对应函数的内容即可
2. 关于自定义Plugin的依赖库问题，他主要依赖`TensorRT-OSS/plugin/common`和`TensorRT-(版本号)/samples/common`下的库，因此要将对应版本的`TensorRT-OSS和TensorRT`该路径下的文件全部复制到一个文件夹下(如本repo中的common文件夹内)，然后使用cmake编译生成对应的库，并将该库链接到`einsum.cpp`即可