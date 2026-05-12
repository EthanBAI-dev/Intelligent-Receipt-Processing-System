#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

// 找到最大轮廓
static std::vector<cv::Point> biggestContour(const std::vector<std::vector<cv::Point>>& contours, double& maxArea) {
	std::vector<cv::Point> biggest;
	maxArea = 0.0;

	for (const auto& contour : contours) {
		double area = cv::contourArea(contour);
		if (area > maxArea) {
			maxArea = area;
			biggest = contour;
		}
	}
	return biggest;
}

// 重新排序点 (左上，右上，左下，右下)
static std::vector<cv::Point> reorder(const std::vector<cv::Point>& points) {
	std::vector<cv::Point> result = points;
	std::vector<std::pair<int, int>> sumDif;

	for (const auto& p : points) {
		sumDif.push_back({ p.x + p.y, p.x - p.y });
	}

	// 左上: 和最小
	auto minSum = std::min_element(sumDif.begin(), sumDif.end(),
		[](const auto& a, const auto& b) { return a.first < b.first; });
	result[0] = points[std::distance(sumDif.begin(), minSum)];

	// 右下: 和最大
	auto maxSum = std::max_element(sumDif.begin(), sumDif.end(),
		[](const auto& a, const auto& b) { return a.first < b.first; });
	result[3] = points[std::distance(sumDif.begin(), maxSum)];

	// 右上: 差最大
	auto maxDiff = std::max_element(sumDif.begin(), sumDif.end(),
		[](const auto& a, const auto& b) { return a.second < b.second; });
	result[1] = points[std::distance(sumDif.begin(), maxDiff)];

	// 左下: 差最小
	auto minDiff = std::min_element(sumDif.begin(), sumDif.end(),
		[](const auto& a, const auto& b) { return a.second < b.second; });
	result[2] = points[std::distance(sumDif.begin(), minDiff)];

	return result;
}

// 绘制矩形
static void drawRectangle(cv::Mat img, const std::vector<cv::Point>& points, int thickness) {

	cv::line(img, points[0], points[1], cv::Scalar(0, 255, 0), thickness);
	cv::line(img, points[1], points[3], cv::Scalar(0, 255, 0), thickness);
	cv::line(img, points[3], points[2], cv::Scalar(0, 255, 0), thickness);
	cv::line(img, points[0], points[2], cv::Scalar(0, 255, 0), thickness);

}

int main() 
{
	Mat src = imread("imgWarp.jpg", IMREAD_COLOR);
	//step1 图像预处理
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	Mat bin;
	double thresh = threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	//step2 轮廓查找及最大轮廓
	Mat imgContours = src.clone();
	Mat imgBigContours = src.clone();

	std::vector<std::vector<Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(bin, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	cv::drawContours(imgContours, contours, -1, Scalar(0, 255, 0), 2);
	double maxArea;
	vector<Point> biggest = biggestContour(contours, maxArea);

	int h = 640;
	int w = 480;
	Mat imgWarp;
	if (!biggest.empty()) 
	{
		//step3 多边形近似及角点排序
		approxPolyDP(biggest, biggest, 0.02 * arcLength(biggest,true), true);
		biggest = reorder(biggest);
		drawRectangle(imgBigContours, biggest, 2);
		//step4 透视变换
		vector<Point2f> pts1(biggest.begin(), biggest.end());
		std::vector<Point2f> pts2{ Point2f{0,0},Point2f{float(w),0},Point2f{0,float(h)},Point2f{float(w),float(h)} };

		Mat matrix = getPerspectiveTransform(pts1, pts2);
		warpPerspective(src, imgWarp, matrix, Size(w, h));
	}
	namedWindow("src", WINDOW_NORMAL);
	imshow("src", imgBigContours);
	namedWindow("imgWarp", WINDOW_NORMAL);
	imshow("imgWarp", imgWarp);
	waitKey(0);
	return 0;

}