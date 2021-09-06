// ���� ifdef ���Ǵ���ʹ�� DLL �������򵥵�
// ��ı�׼�������� DLL �е������ļ��������������϶���� ZY_DL_FORBROKEN_EXPORTS
// ���ű���ġ���ʹ�ô� DLL ��
// �κ�������Ŀ�ϲ�Ӧ����˷��š�������Դ�ļ��а������ļ����κ�������Ŀ���Ὣ
// ZY_DL_FORBROKEN_API ������Ϊ�Ǵ� DLL ����ģ����� DLL ���ô˺궨���
// ������Ϊ�Ǳ������ġ�
#ifdef ZY_DL_FORBROKEN_EXPORTS
#define ZY_DL_FORBROKEN_API __declspec(dllexport)
#else
#define ZY_DL_FORBROKEN_API __declspec(dllimport)
#endif

#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>

//// �����Ǵ� Zy_DL_ForBroken.dll ������
////class ZY_DL_FORBROKEN_API CZy_DL_ForBroken {
////public:
////	CZy_DL_ForBroken(void);
////	// TODO:  �ڴ�������ķ�����
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
