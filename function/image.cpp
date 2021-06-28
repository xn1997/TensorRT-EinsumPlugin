#include <opencv2/opencv.hpp>
#include "image.hpp"

static const float kMean[3] = { 0.485f, 0.456f, 0.406f };
static const float kStdDev[3] = { 0.229f, 0.224f, 0.225f };
static const int map_[7][3] = { {0,0,0} ,
				{128,0,0},
				{0,128,0},
				{0,0,128},
				{128,128,0},
				{128,0,128},
				{0,128,0}};


float* normal(cv::Mat img) {
    //将cv::Mat格式的图片,转换成一维float向量
    float * data = (float*)calloc(img.rows*img.cols * img.channels(), sizeof(float));
//    printf("image channel %d\n",img.channels());
    if(img.channels()==3){
        for (int c = 0; c < 3; ++c)
        {
            for (int i = 0; i < img.rows; ++i)
            { //获取第i行首像素指针
                cv::Vec3b *p1 = img.ptr<cv::Vec3b>(i);
                for (int j = 0; j < img.cols; ++j)
                {
                    data[c * img.cols * img.rows + i * img.cols + j] = (p1[j][c] / 255.0f - kMean[c]) / kStdDev[c];
                }
            }
        }
    }
    else if(img.channels()==1){
        for (int c = 0; c < 1; ++c)
        {
            for (int i = 0; i < img.rows; ++i)
            { //获取第i行首像素指针
                cv::Vec<uchar,1> *p1 = img.ptr<cv::Vec<uchar,1>>(i);
                for (int j = 0; j < img.cols; ++j)
                {
                    data[c * img.cols * img.rows + i * img.cols + j] = p1[j][c];
                }
            }
        }
    }
    else{
        printf("!!!!!!!!!!!!!!!!!!图片输入错误\n");
    }
	return data;
}
