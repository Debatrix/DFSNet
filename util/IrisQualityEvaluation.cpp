#include "IrisQualityEvaluation.h"

#include <io.h>

#ifndef maximum
#define maximum(a,b)    (((a) > (b)) ? (a) : (b))
#endif

#ifndef minimum
#define minimum(a,b)    (((a) < (b)) ? (a) : (b))
#endif

//*************************************************************************************************************************
double qeFocusMeasure(cv::Mat &srcImage, cv::Rect Roi, int downsample_factor)
{
    assert(downsample_factor > 0);
    assert(srcImage.channels() == 1);

    cv::Mat roiImage, gradientX, gradientY;

    if (Roi.area() > 0)
    {
        srcImage(Roi).copyTo(roiImage);
    }
    else
    {
        srcImage.copyTo(roiImage);
    }
    
    if (downsample_factor != 1)
    {
        double resize_ratio = 1 / (double)downsample_factor;
        cv::resize(roiImage, roiImage, cv::Size(), resize_ratio, resize_ratio);
    }

    //有一定帮助 但是速度不可接受
    //cv::equalizeHist(roiImage, roiImage);

    cv::Sobel(roiImage, gradientX, CV_32F, 1, 0);
    cv::Sobel(roiImage, gradientY, CV_32F, 0, 1);

    cv::pow(gradientX, 2, gradientX);
    cv::pow(gradientY, 2, gradientY);
    roiImage = gradientX + gradientY;
    double fm = -1;
    cv::sqrt(roiImage, roiImage);
    fm = cv::mean(roiImage)[0];

    roiImage.release();
    gradientX.release();
    gradientY.release();

    return fm;
}

cv::Rect qeFaceLocation(cv::Mat &srcImage, int threshold, int downsample_factor, double ystart, double hrange)
{

    assert(downsample_factor > 0);
    assert(srcImage.channels() == 1);

    int width = srcImage.cols;
    int height = srcImage.rows;

    int x_range = int(width / downsample_factor);
    int y_range = int(height / downsample_factor);

    std::vector<int> w_sum;
    std::vector<int> h_sum;

    for (int x = 0; x < x_range; x++)
    {
        w_sum.push_back(0);
    }
    for (int y = 0; y < y_range; y++)
    {
        h_sum.push_back(0);
    }

	double color_count[256];
	for (int i = 0; i < 256; i++)
	{
		color_count[i] = 0;
	}
	int value = 0;
	for (int x = 0; x < x_range; x++)
	{
		for (int y = 0; y < y_range; y++)
		{
			value = srcImage.data[(y * downsample_factor) * width + (x * downsample_factor)];
			color_count[value] += 1;
		}
	}
	

    // 分别沿xy轴统计亮度大于阈值的像素数
    //int value = 0;
    for (int x = 0; x < x_range; x++)
    {
        for (int y = 0; y < y_range; y++)
        {
            value = srcImage.data[(y * downsample_factor) * width + (x * downsample_factor)];
            //value = int(srcImage.at<uchar>(y * downsample_factor, x * downsample_factor));
            if (value > threshold)
            {
                w_sum[x] += 1;
                h_sum[y] += 1;
            }
        }
    }

    // 搜索选择区域范围
    int w_start = 0;
    int w_end = 0;
    for (int x = 1; x < x_range; x++)
    {
        if (w_sum[x] > 0)
        {
            w_start = x;
            break;
        }
    }
    for (int x = x_range - 1; x >= 0; x--)
    {
        if (w_sum[x] > 0)
        {
            w_end = x;
            break;
        }
    }
    int h_start = 0;
    int h_end = 0;
    for (int y = 1; y < y_range; y++)
    {
        if (h_sum[y] > 0)
        {
            h_start = y;
            break;
        }
    }
    for (int y = y_range - 1; y >= 0; y--)
    {
        if (h_sum[y] > 0)
        {
            h_end = y;
            break;
        }
    }

    // 计算roi
    int roi_x = w_start * downsample_factor;
    int roi_w = (w_end - w_start) * downsample_factor;
    int roi_y = int((h_start + (h_end - h_start) * ystart) * downsample_factor);
    int roi_h = int((h_end - h_start) * hrange * downsample_factor);

    cv::Rect roi = cv::Rect(roi_x, roi_y, roi_w, roi_h);

    return roi;
}

void qeMaskDenoise(cv::Mat &srcMask)
{
    assert(srcMask.channels() == 1);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(srcMask, srcMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(srcMask, srcMask, cv::MORPH_CLOSE, kernel);

    kernel.release();
}

cv::Rect qeIrisLocation(cv::Mat &srcIris)
{
    int width = srcIris.cols;
    int height = srcIris.rows;

    cv::Rect roi = qeFaceLocation(srcIris, 128, 4, 0, 1);
    //roi = roi + cv::Point(-8, -5);
    //roi = roi + cv::Size(16, 10);

    return roi;
}

double qeImageEntropy(cv::Mat& srcImage, int downsample_factor)
{

	assert(srcImage.channels() == 1);

	double ent = 0.0;
	int imgValue = 0;
	double p = 0.0;

	int width = srcImage.cols;
	int height = srcImage.rows;

	int x_range = int(width / downsample_factor);
	int y_range = int(height / downsample_factor);

	double color_count[256];
	for (int i = 0; i < 256; i++)
	{
		color_count[i] = 0;
	}

	for (int x = 0; x < x_range; x++)
	{
		for (int y = 0; y < y_range; y++)
		{
			imgValue = srcImage.data[(y * downsample_factor) * width + (x * downsample_factor)];
			color_count[imgValue] += 1;
		}
	}
	for (int i = 0; i < 256; i++)
	{
		p = color_count[i] / (x_range * y_range);
		if (p > 0)
		{
			ent -= p * log(p);
		}

	}

	return ent;
}


void qeMinAreaRect(cv::Mat& srcMask)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat binaryImage;
    cv::threshold(srcMask, binaryImage, 100, 255, CV_THRESH_BINARY_INV);
    cv::findContours(binaryImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point());
    cv::RotatedRect rect = cv::minAreaRect(contours);

}


std::vector<double> qeIrisQuality(cv::Mat &srcImage, cv::Mat &srcMask, cv::Mat &srcIris, cv::Mat &srcPupil)
{
    //vector<double> &quality[focus measure, gray, usable aera, dilation, iris shape, pupil shape]

    assert(srcImage.channels() == 1);
    assert(srcMask.channels() == 1);
    assert(srcIris.channels() == 1);
    assert(srcPupil.channels() == 1);

    cv::Mat roiImage, roiMask, roiIris, roiPupil;
	// cv::Mat gradientX, gradientY, gradient;

    srcIris.copyTo(roiIris);
    qeMaskDenoise(roiIris);
    cv::Rect IrisRect = qeIrisLocation(roiIris);
    double qeIrisradius = sqrt(pow(IrisRect.tl().x - IrisRect.br().x, 2) + pow(IrisRect.tl().y - IrisRect.br().y, 2)) / 2;
    cv::Point IrisCenter;
    IrisCenter.x = (IrisRect.tl().x + IrisRect.br().x) / 2;
    IrisCenter.y = (IrisRect.tl().y + IrisRect.br().y) / 2;
    
    srcPupil.copyTo(roiPupil);
    qeMaskDenoise(roiPupil);
    cv::Rect PupilRect = qeIrisLocation(roiPupil);
    cv::Point PupilCenter;
    PupilCenter.x = (PupilRect.tl().x + PupilRect.br().x) / 2;
    PupilCenter.y = (PupilRect.tl().y + PupilRect.br().y) / 2;

    double qeConcentricity = 1 - (sqrt(pow(IrisCenter.x - PupilCenter.x, 2) + pow(IrisCenter.y - PupilCenter.y, 2)) / qeIrisradius);
    
    // Margin adequacy
    double LM = (IrisCenter.x - qeIrisradius) / qeIrisradius;
    double RM = (srcImage.rows - (IrisCenter.x + qeIrisradius)) / qeIrisradius;
    double UM = (IrisCenter.y - qeIrisradius) / qeIrisradius;
    double DM = (srcImage.cols - (IrisCenter.y + qeIrisradius)) / qeIrisradius;
    double LEFT_MARGIN = maximum(0, minimum(1, LM / 0.6));
    double RIGHT_MARGIN = maximum(0, minimum(1, RM / 0.6));
    double UP_MARGIN = maximum(0, minimum(1, UM / 0.2));
    double DOWN_MARGIN = maximum(0, minimum(1, DM / 0.2));
    double qeMargin = minimum(minimum(LEFT_MARGIN, RIGHT_MARGIN), minimum(UP_MARGIN, DOWN_MARGIN));

    srcImage(IrisRect).copyTo(roiImage);
    srcMask(IrisRect).copyTo(roiMask);
    srcIris(IrisRect).copyTo(roiIris);
    srcPupil(IrisRect).copyTo(roiPupil);


    const int width = roiImage.cols;
    const int height = roiImage.rows;


    double color_count[256];
    for (int i = 0; i < 256; i++) 
    { 
        color_count[i] = 0;
    }

    double num_mask_pix = 0;
    double num_iris_pix = 0;
    double num_pupil_pix = 0;
    double focus_count = 0;

    int imgValue = 0;
    int maskValue = 0;
    int irisValue = 0;
    int pupilValue = 0;
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            imgValue = roiImage.data[y * width + x];
            maskValue = roiMask.data[y * width + x];
            irisValue = roiIris.data[y * width + x];
            pupilValue = roiPupil.data[y * width + x];

            if (irisValue > 0)
            {
                num_iris_pix += 1;
                if (pupilValue < 1 && maskValue > 0)
                {
                    num_mask_pix += 1;
                    color_count[imgValue] += 1;
                }
            }
            if (pupilValue > 0)
            {
                num_pupil_pix += 1;
            }

        }
    }

    double p = 0;
    double ent = 0.0;
    for (int i = 0; i < 256; i++)
    {
        p = color_count[i] / num_mask_pix;
        if (p > 0)
        {
            ent -= p * log(p);
        }

    }

    std::vector<double> quality;

    quality.push_back(qeFocusMeasure(srcImage, IrisRect, 1));
    quality.push_back(ent);
    quality.push_back(num_mask_pix /(num_iris_pix - num_pupil_pix));
    quality.push_back(sqrt(num_pupil_pix / num_iris_pix));
    quality.push_back(qeIrisradius);
    quality.push_back(qeConcentricity);
    quality.push_back(qeMargin);


    roiImage.release();
    roiMask.release();
    roiIris.release();
    roiPupil.release();

    return quality;
}

//*************************************************************************************************************************

int qeFaceValidCheck(cv::Mat &srcImage, cv::Rect &face_roi)
{
    int flag = qeSuccess;
    try
    {
        face_roi = qeFaceLocation(srcImage, qeFaceBinarizationThreshold, qeFaceLocationDownsampleFactor, qeFaceLocationYAxisStart, qeFaceLocationHeightRange);

    }
    catch (...)
    {
        flag = qeFaceLocationError;
    }
    if (flag == qeSuccess)
    {
        if (face_roi.area() < qeMinFaceAera) { flag = qeFaceAeraTooSmall; }
        if (face_roi.area() > qeMaxFaceAera) { flag = qeFaceAeraTooLarge; }

		if (qeDebug == 1) { std::cout << face_roi.area() << ';'; }
    }

    return flag;
}

int qeFaceQualityCheck(cv::Mat &srcImage, cv::Rect &face_roi)
{
    int flag = qeSuccess;
    double focus_score = -1;
	double ent = -1;
	try
	{
		focus_score = qeFocusMeasure(srcImage, face_roi, qeFaceFocusMeasureDownsampleFactor);
		if (qeDebug == 1) { std::cout << focus_score << ','; }
	}
	catch (...)
	{
		flag = qeFocusMeasureError;
	}
	if (flag == qeSuccess)
	{
		if (focus_score < qeMaxFaceFocusScore) { flag = qeFaceDefocusBlur; }
		if (focus_score > qeMinFaceFocusScore) { flag = qeFaceMotionBlur; }
	}
	try
	{
		ent = qeImageEntropy(srcImage, 8);
		if (qeDebug == 1) { std::cout << ent << ';'; }
	}
	catch (...)
	{
		flag = qeEntropyCalculateError;
	}
	if (flag == qeSuccess)
	{
		if (ent < qeMinFaceEntropy) { flag = qeFaceLowEntropy; }
	}
    return flag;
}

int qeIrisQualityCheck(cv::Mat &srcImage, cv::Mat &srcMask, cv::Mat &srcIris, cv::Mat &srcPupil)
{
    int flag = qeSuccess;
	std::vector<double> quality;
    try
    {
        quality = qeIrisQuality(srcImage, srcMask, srcIris, srcPupil);
		if (qeDebug == 1) { 
            std::cout << quality[0] << ',' << quality[1] << ',' << quality[2] << ',' << quality[3] << ',' << quality[4] << ',';
            std::cout << quality[5] << ';' << quality[6] << ';';
        }
    }
    catch (...)
    {
        flag = qeFocusMeasureError;
    }
    if (flag == qeSuccess)
    {   
        if (quality[4] < qeMinIrisradius) { return qeIrisradiusTooSmall; }
		if (quality[2] > qeMaxIrisUsableAera) { return qeIrisSegmentError; }
		if (quality[2] < qeMinIrisUsableAera) { return qeIrisAeraTooSmall; }
		if (quality[1] < qeMinIrisEntropy) { return qeIrisLowEntropy; }
		if (quality[0] < qeMinIrisFocusScore) { return qeIrisDefocusBlur; }
		if (quality[0] > qeMaxIrisFocusScore) { return qeIrisMotionBlur; }
		if (quality[3] > qeMaxPupilDilation) { return qePupilAeraTooBig; }
		if (quality[3] < qeMinPupilDilation) { return qePupilAeraTooSmall; }
        if (quality[5] < qeMinConcentricity) { return qeConcentricityTooSmall; }
        if (quality[6] < qeMinMargin) { return qeMarginTooSmall; }
    }

    return flag;
}
