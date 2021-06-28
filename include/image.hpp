#pragma once
typedef struct {
	int w;
	int h;
	int c;
	float *data;
} image;
float* normal(cv::Mat img);
