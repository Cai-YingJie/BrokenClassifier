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

#include"Zy_DL_ForBroken.h"
#include"zyBrokenRecongition.h"

#include<opencv2\opencv.hpp>
//using namespace nsImageRecongition;

nsImageRecongition::imageRecongition::classifier::classifier()
{

}

nsImageRecongition::imageRecongition::classifier::~classifier()
{
	context->destroy();
	engine->destroy();
	runtime->destroy();
}

bool nsImageRecongition::imageRecongition::classifier::initial(const std::string& filePath)
{
	std::cout << "1.0" << std::endl;
	INPUT_H = 160;
	INPUT_W = 160;
	OUTPUT_SIZE = 2;
	BS = 1;

	INPUT_BLOB_NAME = "data";
	OUTPUT_BLOB_NAME = "prob";

	data = new float[3 * INPUT_H * INPUT_W];
	prob = new float[OUTPUT_SIZE];

	
	//char *trtModelStream{ nullptr };
	//size_t size{ 0 };
	std::cout << "2.0" << std::endl;
	std::cout << "filePath :" << filePath << std::endl;
	std::ifstream file(filePath, std::ios::binary);
	if (file.good()) {
		std::cout << "3.0" << std::endl;
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		assert(trtModelStream);
		file.read(trtModelStream, size);
		file.close();
		std::cout << "4.0" << std::endl;
	}
	std::cout << "4.1" << std::endl;
	/*IRuntime**/ runtime = createInferRuntime(gLogger);
	std::cout << "4.2" << std::endl;
	assert(runtime != nullptr);
	std::cout << "5.0" << std::endl;
	/*ICudaEngine**/ engine = runtime->deserializeCudaEngine(trtModelStream, size);
	assert(engine != nullptr);
	std::cout << "6.0" << std::endl;
	/*IExecutionContext**/ context = engine->createExecutionContext();
	assert(context != nullptr);
	delete[] trtModelStream;
	std::cout << "7.0" << std::endl;
	std::cout << "8.0" << std::endl;
	return true;
}

void nsImageRecongition::imageRecongition::classifier::doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();

	// Pointers to input and output device buffers to pass to engine.
	// Engine requires exactly IEngine::getNbBindings() number of buffers.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// Create GPU buffers on device
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	// Create stream
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

int nsImageRecongition::imageRecongition::classifier::predict(const cv::Mat& inputImg)
{

	cv::Mat resizeImg;
	//cv::imshow("", inputImg);
	//cvWaitKey(0);
	cv::resize(inputImg, resizeImg, cv::Size(INPUT_H, INPUT_W),/*CV_INTER_NN*/CV_INTER_LINEAR);
	//cv::imshow("", resizeImg);
	//cvWaitKey(0);
	int i = 0;
	int b = 0;
	for (int row = 0; row < INPUT_H; ++row) {
		uchar* uc_pixel = inputImg.data + row * inputImg.step;
		for (int col = 0; col < INPUT_W; ++col) {
			data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
			data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
			data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}
	//for (int i = 0; i < INPUT_H * INPUT_W; i++)
	//{
	//	std::cout << data[i] << std::endl;
	//}
	auto start = std::chrono::system_clock::now();
	doInference(*context, data, prob, 1);
	auto end = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	// Destroy the engine


	for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
	{
		std::cout << prob[i] << ", ";
		//if (i % 10 == 0) std::cout << i / 10 << std::endl;
	}
	if (prob[0] > prob[1])
	{
		return 0;
	}
	else
	{
		return 1;
	}
	//return 0;
}
