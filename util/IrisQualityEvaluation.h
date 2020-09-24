#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <assert.h>
#include <cmath>

//***************************************** 参数设定 **************************************************************

//控制设置
#define qeDebug 1

//常规参数
#define qeFaceBinarizationThreshold 180
#define qeFaceLocationDownsampleFactor 16
#define qeFaceLocationYAxisStart 0.15
#define qeFaceLocationHeightRange 0.5
#define qeFaceFocusMeasureDownsampleFactor 2

//阈值
#define qeMinFaceAera 500000
#define qeMaxFaceAera 6266880
#define qeMinFaceEntropy 3.4
#define qeMaxFaceFocusScore 38.0
#define qeMinFaceFocusScore 20.0
#define qeMaxIrisFocusScore 47.0
#define qeMinIrisFocusScore 28.0
#define qeMinIrisEntropy 6
#define qeMaxIrisUsableAera 1.0
#define qeMinIrisUsableAera 0.7
#define qeMaxPupilDilation 0.7
#define qeMinPupilDilation 0.2
#define qeMinIrisradius 80
#define qeMinConcentricity 0.9
#define qeMinMargin 0.8

//********************************************* 错误代码 **************************************************************

//正常
#define qeSuccess 0

//运行错误
#define qeFocusMeasureError -1001
#define qeFaceLocationError -1002
#define qeEntropyCalculateError -1003

//低质量图像
#define qeFaceAeraTooSmall -1011
#define qeFaceAeraTooLarge -1012
#define qeFaceLowEntropy -1013
#define qeFaceDefocusBlur -1021
#define qeFaceMotionBlur -1022
#define qeIrisDefocusBlur -1023
#define qeIrisMotionBlur -1024
#define qeIrisLowEntropy -1031
#define qeIrisSegmentError -1032
#define qeIrisAeraTooSmall -1033
#define qePupilAeraTooSmall -1034
#define qePupilAeraTooBig -1035
#define qeIrisradiusTooSmall -1036
#define qeConcentricityTooSmall -1037
#define qeMarginTooSmall -1038

/*********************************************************************************************************
函数类型：int
函数参数：
输入
cv::Mat& srcImage
含义：单通道脸部图像

输出
cv::Rect& face_roi
含义：可能含有眼部的区域,大小不定

返回值类型：int
含义：状态值,参见错误代码

函数功能：排除不含有效人脸或人脸面积过大的无效图像,并返回眼部区域粗估计结果.
**********************************************************************************************************/
int qeFaceValidCheck(cv::Mat& srcImage, cv::Rect& face_roi);

/*********************************************************************************************************
函数类型：int
函数参数：
输入
1.cv::Mat& srcImage
含义：单通道脸部图像

2.cv::Rect& face_roi
含义：可能含有眼部的区域,大小不定

输出

返回值类型：int
含义：状态值,参见错误代码

函数功能：排除脸部区域模糊和欠曝/过曝图像
**********************************************************************************************************/
int qeFaceQualityCheck(cv::Mat& srcImage, cv::Rect& face_roi);

/*********************************************************************************************************
函数类型：int
函数参数：
输入
1. cv::Mat &srcImage
含义：单通道眼部图像

2. cv::Mat &srcMask
含义：单通道mask,0为黑色(遮挡),255为白色

3. cv::Mat &srcIris
含义：单通道Iris mask,0为黑色(遮挡),255为白色

4. cv::Mat &srcPupil
含义：单通道Pupil mask,0为黑色(遮挡),255为白色

输出

返回值类型：int
含义：状态值,参见错误代码

函数功能：排除虹膜区域模糊,欠曝/过曝,有效面积比小和瞳孔缩放过度图像
备注: 输入的图像需要为原始比例(不能被缩放为方形)
**********************************************************************************************************/
int qeIrisQualityCheck(cv::Mat& srcImage, cv::Mat& srcMask, cv::Mat& srcIris, cv::Mat& srcPupil);



/*********************************************************************************************************
函数类型：double
函数参数：
输入
1. cv::Mat &srcImage
含义：原始图像，未经过缩放等操作。

2.cv::Rect Roi
含义：进行质量评价的区域。

3.int downsample_factor
含义：图像缩小的倍数，不应当小于1。

输出
返回值类型：double
含义：模糊度分数。

函数功能：
计算图像离焦模糊的程度。

备注:图像中包含大量头发区域和运动模糊等因素会导致得分过高。
**********************************************************************************************************/
double qeFocusMeasure(cv::Mat &srcImage, cv::Rect Roi, int downsample_factor = 2);

/*********************************************************************************************************
函数类型：cv::Rect
函数参数：
输入
1. cv::Mat &srcImage
含义：原始图像，未经过缩放等操作。

2.uchar threshold
含义：二值化阈值，实验测试80可以有效确定脸部区域。

3.double downsample_factor
含义：图像缩小的倍数，不应当小于1。计算时是隔downsample_factor行/列遍历像素统计，设置到16以上可以提升性能。

4.double ystart, double hrange
含义：roi左上角下移比例和高度缩小比例，用于调整roi区域位置和大小。ystart=0.15,hrange=0.5可以有效确定脸部区域。

输出

1. cv::Rect &roi
含义：脸部区域

返回值类型：cv::Rect
含义：脸部区域roi.

函数功能：
根据图像二值化结果粗估计人脸位置和大小。
备注：当ystart=0.15,hrange=0.5时，roi尺寸小于800*400时，可认为图像中不包含有效人脸。
**********************************************************************************************************/
cv::Rect qeFaceLocation(cv::Mat &srcImage, int threshold = 180, int downsample_factor = 16, double ystart = 0.15, double hrange = 0.5);

/*********************************************************************************************************
函数类型：std::vector<double>
函数参数：
输入
1. cv::Mat &srcImage
含义：单通道眼部图像

2. cv::Mat &srcMask
含义：单通道mask,0为黑色(遮挡),255为白色

3. cv::Mat &srcIris
含义：单通道Iris mask,0为黑色(遮挡),255为白色

4. cv::Mat &srcPupil
含义：单通道Pupil mask,0为黑色(遮挡),255为白色

输出

返回值类型：std::vector<double>
含义：质量分数,共7个元素,分别为虹膜区域清晰度,虹膜区域灰度分布,虹膜区域有效面积比,瞳孔缩放度，虹膜半径，虹膜-瞳孔同心度，边界余量
注:输入的vector<double> &quality必须为空

函数功能：在虹膜定位分割的基础上对虹膜进行质量评价

备注：质量分数可能会增加其他元素,但已有的顺序不变
**********************************************************************************************************/
std::vector<double> qeIrisQuality(cv::Mat &srcImage, cv::Mat &srcMask, cv::Mat &srcIris, cv::Mat &srcPupil);

/*********************************************************************************************************
函数类型：double
函数参数：
输入
1. cv::Mat &srcImage
含义：单通道图像

2.int downsample_factor
含义：图像缩小的倍数，不应当小于1。

输出

返回值类型：double
含义：图像熵值

函数功能：评估整个图像的灰度(亮度)分布.
**********************************************************************************************************/
double qeImageEntropy(cv::Mat& srcImage, int downsample_factor = 1);

/*********************************************************************************************************
函数类型：void
函数参数：
输入
cv::Mat &srcMask
含义：单通道掩膜图像

输出
无

函数功能：去除掩膜的噪点与填补小的漏洞,通过一次开运算与闭运算完成.

注:它是原地修改的.
**********************************************************************************************************/
void qeMaskDenoise(cv::Mat &srcMask);

/*********************************************************************************************************
函数类型：cv::Rect
函数参数：
输入
cv::Mat &srcIris
含义：单通道图像

输出

返回值类型：cv::Rect
含义：补全的虹膜区域

函数功能：根据掩膜图像确定虹膜区域,减少计算量.
**********************************************************************************************************/
cv::Rect qeIrisLocation(cv::Mat &srcIris);

