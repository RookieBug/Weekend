#include <opencv2\opencv.hpp>

using namespace cv;
int value = 1;
int d = 0;
int sigmaX = 0, sigmaY = 0;
int thresholdMin = 20, thresholdMax = 100,size = 3;
Mat cap_img, src_img, gray_img, gaussian_img, canny_img;

Mat acousticImg(Mat img);
Mat dilateErode(Mat src_img);
void gaussianBlurFun(int, void*);
void cannyImg(int, void*);

int main(int argc,char **argv) {
	//读取摄像头
	VideoCapture cap;
	cap = VideoCapture(0);
	
	while (cap.isOpened()) {
		/*读取摄像头数据*/
		cap.read(cap_img);
		/*将图片进行镜像*/
		src_img = acousticImg(cap_img);
		
		/*进行高斯过滤*/
		/*
		namedWindow("GuassianBlur");
		createTrackbar("Gaussian", "GuassianBlur", &value, 5, gaussianBlurFun);
		createTrackbar("SigmaX", "GuassianBlur", &sigmaX, 255, gaussianBlurFun);
		createTrackbar("SigmaY", "GuassianBlur", &sigmaY, 255, gaussianBlurFun);
		*/
		//gaussianBlurFun(0, 0);

		/*将图片转换为灰度图*/
		cvtColor(src_img, gray_img, COLOR_BGR2GRAY);

		/*边缘检测*/
		/*
		namedWindow("Canny");
		createTrackbar("Min", "Canny", &thresholdMin, thresholdMax, cannyImg);
		createTrackbar("Max", "Canny", &thresholdMax, 1000, cannyImg);
		createTrackbar("Size", "Canny", &size, 7, cannyImg);
		cannyImg(0, 0);
		*/

		/*腐蚀减膨胀*/
		Mat res = dilateErode(gray_img);
		/*显示*/
		imshow("Capture", cap_img);
		//imshow("Canny", ~canny_img);
		imshow("Result", ~res);
		if (waitKey(10) == 27) {
			break;
		}
	}
	cap.release();
	destroyAllWindows();
	return 0;
}
/*高斯过滤*/
void gaussianBlurFun(int, void*) {
	GaussianBlur(src_img, gaussian_img, Size(2 * value + 1, 2 * value + 1), sigmaX, sigmaY);
//	imshow("GuassianBlur", gaussian_img);
}
/*边缘检测*/
void cannyImg(int, void*) {
	Canny(gray_img, canny_img, thresholdMin, thresholdMax, size);
}
/*镜像图像*/
Mat acousticImg(Mat img) {
	Mat dst;
	Mat map_x = Mat(cap_img.size(), CV_32FC1);
	Mat map_y = Mat(cap_img.size(), CV_32FC1);
	
	int rows = cap_img.rows;
	int cols = cap_img.cols;
	for (int row = 0;row < rows;row++) {
		for (int col = 0;col < cols;col++) {
			map_x.at<float>(row, col) = static_cast<float>(cols - col);
			map_y.at<float>(row, col) = static_cast<float>(row);
		}
	}
	remap(img, dst, map_x, map_y, CV_INTER_LINEAR);
	return dst;

}

/*计算腐蚀减去膨胀后的图像*/
Mat dilateErode(Mat src_img) {
	int element_size = 3;
	int s = 2 * element_size + 1;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(s, s));
	Mat dilate_img, erode_img;
	dilate(src_img, dilate_img, kernel);
	erode(src_img, erode_img, kernel);

	//将腐蚀和膨胀后的图片进行相减。
	int rows = src_img.rows;
	int cols = src_img.cols;
	Mat result_img = Mat::zeros(src_img.size(), src_img.type());
	for (int row = 0;row < rows;row++) {
		for (int col = 0;col < cols;col++) {
			if (src_img.channels() == 1) {
				result_img.at<uchar>(row, col) = saturate_cast<uchar>(dilate_img.at<uchar>(row, col) - erode_img.at<uchar>(row, col));
			}
			else if (src_img.channels() == 3) {
				result_img.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(dilate_img.at<Vec3b>(row, col)[0] - erode_img.at<Vec3b>(row, col)[0]);
				result_img.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(dilate_img.at<Vec3b>(row, col)[1] - erode_img.at<Vec3b>(row, col)[1]);
				result_img.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(dilate_img.at<Vec3b>(row, col)[2] - erode_img.at<Vec3b>(row, col)[2]);
			}
		}
	}
	return result_img;
}
