// Zy_DL_ForBroken.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "Zy_DL_ForBroken.h"
#include "zyBrokenRecongition.h"

// 这是导出变量的一个示例
ZY_DL_FORBROKEN_API int nZy_DL_ForBroken=0;

// 这是导出函数的一个示例。
ZY_DL_FORBROKEN_API int fnZy_DL_ForBroken(void)
{
    return 42;
}

//// 这是已导出类的构造函数。
//// 有关类定义的信息，请参阅 Zy_DL_ForBroken.h
//CZy_DL_ForBroken::CZy_DL_ForBroken()
//{
//    return;
//}


nsImageRecongition::imageRecongition::imageRecongition()
{
	brokenClassifier = new classifier();
}

nsImageRecongition::imageRecongition::~imageRecongition()
{
	delete brokenClassifier;
}

bool nsImageRecongition::imageRecongition::initial(const std::string& filePath)
{
	return brokenClassifier->initial(filePath);
}

int nsImageRecongition::imageRecongition::detect(const cv::Mat& inputImg)
{
	return brokenClassifier->predict(inputImg);
}