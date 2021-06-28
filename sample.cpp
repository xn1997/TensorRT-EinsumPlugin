#include "Head.h"

using namespace std;
using namespace nvinfer1;

std::string image_path = "../data/tabby_tiger_cat.jpg";
std::string engine_path = "../data/resnet50.engine";
// 0. 参数
int max_batch_size = 1;
int INPUT_H=224,INPUT_W=224,INPUT_C=3;
std::vector<std::vector<int>> OUTPUT_SIZE{{7,7},{1,1000}};  // for reshape output
//std::vector<std::vector<int>> OUTPUT_SIZE{{1,1000}};

int main()
{
    std::unique_ptr<ICudaEngine, InferDeleter> engine(loadEngine(engine_path));
    std::unique_ptr<IExecutionContext, InferDeleter> context(engine->createExecutionContext());

    // 3.1 构建stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 3.2 分配输入、输出内存（cpu+gpu）
    std::vector<int> inputIndex;  // 输入索引
    CPU_data input_cpu_data, output_cpu_data;
    int NbBindings = engine->getNbBindings();  // number of input+output
    void* buffers[NbBindings];
    allocate_buffers(engine,max_batch_size,inputIndex,input_cpu_data,output_cpu_data,buffers);  // initialize buffers(for gpu data)

    // 3.3 加载输入
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    cv::Mat dst = cv::Mat::zeros(INPUT_H, INPUT_W, CV_32FC3);
    cv::resize(img,dst, dst.size());
    float* fileData=normal(dst);
    for(int i = 0; i < INPUT_H*INPUT_W*INPUT_C; ++i){
        input_cpu_data[0][i] = fileData[i];
        //        std::cout << (" .:-=+*#%@"[static_cast<uchar>(fileData[i]) / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");
    }
    free(fileData);  // 释放图片
    // infer
    for(int i = 0; i <2;i++){  // 第一次启动GPU，会消耗很长时间，第二次就正常速度了
        trt_infer(stream,context,max_batch_size,inputIndex,input_cpu_data,output_cpu_data,buffers);
    }

    cudaStreamDestroy(stream);
    //output postprogress
    for(int i = 0; i < OUTPUT_SIZE.size(); ++i){
        std::vector<std::vector<float>> a = reshape_1to2D(OUTPUT_SIZE[i],output_cpu_data[i]);
        printfVector2D(a,10);
    }

    return 1;
}
