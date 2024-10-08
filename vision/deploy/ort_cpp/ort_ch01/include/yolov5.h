#ifndef YOLOV5_H
#define YOLOV5_H

#include "main.h"

// using namespace std;
// using namespace cv;
// using namespace Ort;

struct Net_config
{
    float confThreshold; // Confidence threshold
    float nmsThreshold;  // Non-maximum suppression threshold
    float objThreshold;  // Object Confidence threshold
    string modelpath;    // model path
    bool gpu = false;   // using gpu
};

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

int endsWith(string s, string sub);

const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
                                 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
                                 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

const float anchors_1280[4][6] = { {19, 27, 44, 40, 38, 94},{96, 68, 86, 152, 180, 137},{140, 301, 303, 264, 238, 542},
                       {436, 615, 739, 380, 925, 792} };

class YOLO
{
public:
    YOLO(Net_config config);
    void detect(Mat& frame);
private:
    float* anchors;
    int num_stride;
    int inpWidth;
    int inpHeight;
    int nout;
    int num_proposal;
    vector<string> class_names;
    int num_class;
    int seg_num_class;

    float confThreshold;
    float nmsThreshold;
    float objThreshold;
    const bool keep_ratio = true;
    vector<float> input_image_;
    void normalize_(Mat img);
    void nms(vector<BoxInfo>& input_boxes);
    Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);

    Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5s");
    Ort::Session *ort_session = nullptr;
    SessionOptions sessionOptions = SessionOptions();
    vector<const char* > input_names;
    vector<const char* > output_names;
    vector<vector<int64_t>> input_node_dims; // >=1 outputs
    vector<vector<int64_t>> output_node_dims; // >=1 outputs
    std::vector<AllocatedStringPtr> In_AllocatedStringPtr;
    std::vector<AllocatedStringPtr> Out_AllocatedStringPtr;
};

#endif // YOLOV5_H


class YOLOv5
{
public:
	// YOLOv5(Configuration config);
	YOLOv5(Net_config config);
	void detect(Mat& frame);
private:
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	int num_classes;
	string classes[80] = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus",
							"train", "truck", "boat", "traffic light", "fire hydrant",
							"stop sign", "parking meter", "bench", "bird", "cat", "dog",
							"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
							"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
							"skis", "snowboard", "sports ball", "kite", "baseball bat",
							"baseball glove", "skateboard", "surfboard", "tennis racket",
							"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
							"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
							"hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
							"bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
							"remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
							"sink", "refrigerator", "book", "clock", "vase", "scissors",
							"teddy bear", "hair drier", "toothbrush"};
	const bool keep_ratio = true;
	vector<float> input_image_;		// 输入图片
	void normalize_(Mat img);		// 归一化函数
	void nms(vector<BoxInfo>& input_boxes);  
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left); // 预处理

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-6.1"); //初始化环境，
	Session *ort_session = nullptr;    //初始化Session指针选项
	SessionOptions sessionOptions = SessionOptions();  //初始化Session对象
	//SessionOptions sessionOptions;
	vector<char*> input_names;  // 定义一个字符指针vector
	vector<char*> output_names; // 定义一个字符指针vector
	vector<vector<int64_t>> input_node_dims; // >=1 outputs  ，二维vector
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};



