#include "function.h"
using namespace std;
using namespace nvinfer1;
Logger gLogger;
std::vector<std::vector<float>> reshape_1to2D(std::vector<int> shape,std::vector<float> data){
    std::vector<std::vector<float>> output(shape[0]);
    for(int i=0; i<output.size();++i)
        output[i].resize(shape[1]);

    for(int i=0; i<shape[0];++i){
        for(int j=0; j<shape[1]; ++j){
            output[i][j] = data[i*shape[1]+j];
        }
    }
    return output;
}
void printfVector2D(std::vector<std::vector<float>> arrays,int max_lengths){
    for(int i = 0; i < arrays.size() && i<max_lengths; ++i) {
        for(int j = 0; j < arrays[i].size() && j < max_lengths; ++j) {
            std::cout << arrays[i][j] << " ";
        }
        std::cout << "\n";
    }
}
ICudaEngine* loadEngine(const std::string& engine)
{
    std::ifstream engineFile(engine, std::ios::binary);
    if (!engineFile)
    {
        std::cout << "Error opening engine file: " << engine << std::endl;
        return nullptr;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        std::cout << "Error loading engine file: " << engine << std::endl;
        return nullptr;
    }

    std::unique_ptr<IRuntime,InferDeleter> runtime(createInferRuntime(gLogger));

    return runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
}
