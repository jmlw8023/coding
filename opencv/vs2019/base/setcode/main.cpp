#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{

	string imgPath = "E:/source/code/coding/opencv/data/images/milk1.jpg";
	Mat img = imread(imgPath);

	namedWindow("milk", WINDOW_AUTOSIZE);
	imshow("milk", img);



	waitKey(0);
	destroyAllWindows();

	//cout << "cpp and opencv using in vs2019!!" << endl;



	return 0;
}

