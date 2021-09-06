// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 ZY_DL_FORBROKEN_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// ZY_DL_FORBROKEN_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
#ifdef ZY_DL_FORBROKEN_EXPORTS
#define ZY_DL_FORBROKEN_API __declspec(dllexport)
#else
#define ZY_DL_FORBROKEN_API __declspec(dllimport)
#endif

#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>

//// 此类是从 Zy_DL_ForBroken.dll 导出的
////class ZY_DL_FORBROKEN_API CZy_DL_ForBroken {
////public:
////	CZy_DL_ForBroken(void);
////	// TODO:  在此添加您的方法。
////};
//
//extern ZY_DL_FORBROKEN_API int nZy_DL_ForBroken;
//
//ZY_DL_FORBROKEN_API int fnZy_DL_ForBroken(void);


namespace nsImageRecongition
{
	class ZY_DL_FORBROKEN_API imageRecongition
	{
	public:
		imageRecongition();
		~imageRecongition();

		bool initial(const std::string& filePath);
		int  detect(const cv::Mat& inputImg);

	private:

		class classifier;
		classifier *brokenClassifier;
	};
}
