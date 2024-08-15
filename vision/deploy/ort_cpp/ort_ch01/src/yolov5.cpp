#include "yolov5.h"

int endsWith(string s, string sub) {
    return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}

YOLO::YOLO(Net_config config)
{
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->objThreshold = config.objThreshold;

    string classesFile = "./data/coco.names";
    string model_path = config.modelpath;
    std::wstring widestr = std::wstring(model_path.begin(), model_path.end());

    // if (config.gpu) {
    //     OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   //CUDA加速开启
    // }

    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC); //设置图优化类型

    ort_session = new Session(env, widestr.c_str(), sessionOptions);    // 创建会话，把模型加载到内存中

    size_t numInputNodes = ort_session->GetInputCount();            //输入输出节点数量    
    size_t numOutputNodes = ort_session->GetOutputCount();

    for (int i = 0; i < numInputNodes; i++)                         // onnxruntime1.12版本后不能按照从前格式写
    {
        AllocatorWithDefaultOptions allocator;                              // 配置输入输出节点内存
        In_AllocatedStringPtr.push_back(ort_session->GetInputNameAllocated(i, allocator));
        input_names.push_back(In_AllocatedStringPtr.at(i).get());           // 内存
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);   // 类型
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();                     // 输入shape
        input_node_dims.push_back(input_dims);                              // 输入维度信息
    }
    for (int i = 0; i < numOutputNodes; i++)
    {
        AllocatorWithDefaultOptions allocator;
        Out_AllocatedStringPtr.push_back(ort_session->GetOutputNameAllocated(i, allocator));
        output_names.push_back(Out_AllocatedStringPtr.at(i).get());
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }
    this->inpHeight = input_node_dims[0][2];
    this->inpWidth = input_node_dims[0][3];
    this->nout = output_node_dims[0][2];                // 5+classese 85
    this->num_proposal = output_node_dims[0][1];        // 3*(小检测框+中检测框+大检测框） 3*((20*20)+(40*40)+(80*80))

    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) this->class_names.push_back(line);
    this->num_class = class_names.size();

    if (endsWith(config.modelpath, "6.onnx"))           // 判断版本
    {
        anchors = (float*)anchors_1280;
        this->num_stride = 4;
    }
    else
    {
        anchors = (float*)anchors_640;
        this->num_stride = 3;
    }
}

Mat YOLO::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = this->inpHeight;
    *neww = this->inpWidth;
    Mat dstimg;
    if (this->keep_ratio && srch != srcw) {
        float hw_scale = (float)srch / srcw;
        if (hw_scale > 1) {                             // srch>srcw
            *newh = this->inpHeight;
            *neww = int(this->inpWidth / hw_scale);     // set/scale
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);     // resize(nw,nh)
            *left = int((this->inpWidth - *neww) * 0.5);                // 计算padding距离
            copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);  // padding
        }
        else {
            *newh = (int)this->inpHeight * hw_scale;
            *neww = this->inpWidth;
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
            *top = (int)(this->inpHeight - *newh) * 0.5;
            copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
        }
    }
    else {
        resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
    }
    return dstimg;
}

void YOLO::normalize_(Mat img)
{
    //    img.convertTo(img, CV_32F);
    int row = img.rows;
    int col = img.cols;
    this->input_image_.resize(row * col * img.channels());
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];       // HWC to CHW, BGR to RGB，j * 3 + 2 - c即完成转换
                this->input_image_[c * row * col + i * col + j] = pix / 255.0;
            }
        }
    }
}

void YOLO::nms(vector<BoxInfo>& input_boxes)
{
    sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });   // 按照置信度排序， []Lambda 表达式
    vector<float> vArea(input_boxes.size());                            // 记录每个检测框面积
    for (int i = 0; i < int(input_boxes.size()); ++i)                   // 遍历所有检测框
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
            * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }

    vector<bool> isSuppressed(input_boxes.size(), false);               // 记录是否抑制，默认为FALSE
    for (int i = 0; i < int(input_boxes.size()); ++i)                   // 遍历所有检测框
    {
        if (isSuppressed[i]) { continue; }                              // 是否已经判断过
        for (int j = i + 1; j < int(input_boxes.size()); ++j)           // 第二个指针遍历
        {
            if (isSuppressed[j]) { continue; }
            float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

            float w = (max)(float(0), xx2 - xx1 + 1);
            float h = (max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);          // 计算miou

            if (ovr >= this->nmsThreshold)
            {
                isSuppressed[j] = true;                                 // 大于设定的阈值，则抑制
            }
        }
    }
    // return post_nms;
    int idx_t = 0;
    // remove_if()函数 remove_if(beg, end, op) //移除区间[beg,end)中每一个“令判断式:op(elem)获得true”的元素
    input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

void YOLO::detect(Mat& frame)
{
    int newh = 0, neww = 0, padh = 0, padw = 0;     // padh:上下边的padding距离; padw:左右padding的距离
    Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
    this->normalize_(dstimg);
    array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

    auto memory_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Value input_tensor_ = Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());


    vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size()); 
    vector<BoxInfo> generate_boxes;
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();                  // 数组，存放预测数据 [bs,anchor'classes,anchor'number,pos+conf+ num'classes]
    float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;         // 计算缩放倍数
    for (int i = 0; i < num_proposal; ++i)      // 遍历所有的num_pre_boxes 3*((20*20)+(40*40)+(80*80))
    {
        int index = i * nout;                   // 索引
        float obj_conf = pdata[index + 4];      // 第四个为置信度分数
        if (obj_conf > this->objThreshold)      // 大于阈值
        {
            // 求最大分数和索引
            int class_idx = 0;                  // 记录类别id
            float max_class_socre = 0;          // 记录最大概率
            for (int k = 0; k < this->num_class; ++k)   // K个类别里循环
            {
                if (pdata[k + index + 5] > max_class_socre) // 判断分数
                {
                    max_class_socre = pdata[k + index + 5]; // 记录分数
                    class_idx = k;                          // 记录类别数
                }
            }
            max_class_socre *= obj_conf;   // 最大的类别分数*置信度
            if (max_class_socre > this->confThreshold) // 再次筛选
            {
                float cx = pdata[index];        //x：检测框中心点
                float cy = pdata[index + 1];    //y
                float w = pdata[index + 2];     //w：检测框宽
                float h = pdata[index + 3];     //h
                // 映射到原来的图像上
                float xmin = (cx - padw - 0.5 * w)*ratiow;      // （推理位置-左边padding距离-0.5*宽）*缩放倍数=原图像左上角x位置
                float ymin = (cy - padh - 0.5 * h)*ratioh;      // （推理位置-上边padding距离-0.5*高）*缩放倍数=原图像左上角y位置
                float xmax = (cx - padw + 0.5 * w)*ratiow;      // 
                float ymax = (cy - padh + 0.5 * h)*ratioh;      // 

                generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, class_idx }); //记录相关数据
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    nms(generate_boxes);
    for (size_t i = 0; i < generate_boxes.size(); ++i)
    {
        int xmin = int(generate_boxes[i].x1);
        int ymin = int(generate_boxes[i].y1);
        rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
        string label = format("%.2f", generate_boxes[i].score);
        label = this->class_names[generate_boxes[i].label] + ":" + label;
        putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
    }
}
