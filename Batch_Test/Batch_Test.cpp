// Batch_Test.cpp : 定义控制台应用程序的入口点。
//
#define _AFXDLL
#include "stdafx.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include <afxwin.h>
//#include <windows.h>
#include <direct.h>
#include<stdio.h>

#include "Zy_DL_ForBroken.h"

using namespace nsImageRecongition;
using namespace cv;
//using namespace std;

// 遍历文件夹图像并处理
void FindFiles(CString filePath, std::vector<std::string>& fileNameVec)
{
	std::cout << "开始获取文件夹文件名......" << std::endl;
	fileNameVec.clear();

	CString fileName = _T("");
	filePath += _T("\\*.jpg");

	CFileFind finder;
	BOOL bFind = FALSE;
	bFind = finder.FindFile(filePath);
	while (bFind)
	{
		bFind = finder.FindNextFile();
		if (finder.IsDots())
		{
			continue;
		}
		else
		{
			fileName = finder.GetFilePath();
			USES_CONVERSION;
			std::string ss(W2A(fileName));
			fileNameVec.push_back(ss);
		}
	}
	std::cout << std::endl;
	finder.Close();
	std::cout << "结束获取文件夹文件名......" << std::endl;
}

int main()
{
	imageRecongition IR;
	/*CString enginePath("");
	enginePath = _T("E:\\ALGXML\\mobilenetv3small.engine");*/
	CString ngPath;
	CString okPath;

	std::cout << "开始初始化：" << std::endl;
	std::string filePath = "C:\\ALGXML\\BrokenModel.engine";
	//enginePath.c_str();
	std::cout << "路径："<< filePath << std::endl;
	IR.initial(filePath);
	std::cout << "初始化完成：" << std::endl;

	std::vector<std::string> vecFilePath;
	//string filter = "C:\\Users\\Administrator\\Desktop\\img\\broken\\*.jpg";
	CString path("");
	path = _T("C:\\ALGXML\\broken");
	
	ngPath = path + "\\" + "NG";
	okPath = path + '\\' + "OK";

	
	USES_CONVERSION;
	//string strNG;
	std::string strNG(W2A(ngPath));
	//string strOK;
	string strOK(W2A(okPath));

	_mkdir(strNG.c_str());
	_mkdir(strOK.c_str());

	FindFiles(path, vecFilePath);
	Mat src;
	for (int i = 0; i < vecFilePath.size(); i++)
	{
		//string str1 = vecFilePath[i];
		Mat dst;
		src = imread(vecFilePath[i]);
		//cvtColor(src, dst, CV_BGR2GRAY);
		//std::cout << "channels" << src.channels()<< std::endl;
		resize(src, dst, Size(160,160),INTER_LINEAR);

		int result  = IR.detect(dst);

		int pos = vecFilePath[i].rfind("\\");
		string imgName = vecFilePath[i].substr(pos);
		if (result == 0)
		{
			imwrite(strNG + imgName, src);
		}
		else
		{
			imwrite(strOK + imgName, src);
		}
	}
	std::cin.get();
    return 0;
}

