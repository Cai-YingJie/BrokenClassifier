#pragma once
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"

#include "Zy_DL_ForBroken.h"


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


using namespace nvinfer1;

//using namespace nsImageRecongition;


class nsImageRecongition::imageRecongition::classifier
{

public:

	classifier();
	~classifier();

	bool initial(const std::string& filePath);

	int predict(const cv::Mat& inputImg);

private:

	int INPUT_H;

	int INPUT_W;
	int OUTPUT_SIZE;
	int BS;

	const char* INPUT_BLOB_NAME;
	const char* OUTPUT_BLOB_NAME;

	float* data;
	float* prob;

	IExecutionContext* context;
	ICudaEngine* engine;
	IRuntime* runtime;

	Logger gLogger;

	char *trtModelStream;
	size_t size;

	void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
};