// Zy_DL_ForBroken.cpp : ���� DLL Ӧ�ó���ĵ���������
//

#include "stdafx.h"
#include "Zy_DL_ForBroken.h"
#include "zyBrokenRecongition.h"

// ���ǵ���������һ��ʾ��
ZY_DL_FORBROKEN_API int nZy_DL_ForBroken=0;

// ���ǵ���������һ��ʾ����
ZY_DL_FORBROKEN_API int fnZy_DL_ForBroken(void)
{
    return 42;
}

//// �����ѵ�����Ĺ��캯����
//// �й��ඨ�����Ϣ������� Zy_DL_ForBroken.h
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