/**
 * @file opencv_backend.cc
 * @brief OpenCV DNN backend implementation source file
 *
 * Uses OpenCV DNN module for YOLO inference.
 * Requires OpenCV 4.x or higher.
 */

#include "backend/ibackend.hpp"
#include <vector>

#if YOLO_OPENCV_BACKEND
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#endif

namespace yolo {

// ============================================================
// OpenCvBackend Implementation
// ============================================================

bool OpenCvBackend::initialize(const ModelConfig& config) {
    config_ = config;

#if YOLO_OPENCV_BACKEND
    try {
        cv::dnn::Net net = cv::dnn::readNet(config_.model_path);
        if (net.empty()) {
            return false;
        }

        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        std::vector<int> layer_ids = net.getUnconnectedOutLayersIds();
        std::vector<cv::String> layer_names = net.getLayerNames();
        for (int id : layer_ids) {
            output_names_.push_back(layer_names[id - 1]);
        }

        net_ = new cv::dnn::Net(net);
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        return false;
    }
#else
    return false;
#endif
}

std::vector<DetectResult> OpenCvBackend::infer(
    const uint8_t* data,
    int width,
    int height,
    int channels,
    bool is_bgr
) {
    std::vector<DetectResult> results;

#if YOLO_OPENCV_BACKEND
    if (!net_) return results;

    cv::Mat image(height, width, channels == 3 ? CV_8UC3 : CV_8UC1, (void*)data);

    if (is_bgr && channels == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    }

    cv::Mat blob = cv::dnn::blobFromImage(
        image,
        1.0 / 255.0,
        cv::Size(config_.input_width, config_.input_height),
        cv::Scalar(0, 0, 0),
        true,
        false
    );

    cv::dnn::Net* net_ptr = static_cast<cv::dnn::Net*>(net_);
    net_ptr->setInput(blob);
    cv::Mat output = net_ptr->forward();

    int num_detections = output.size[1];
    int num_classes = output.size[2] - 5;

    for (int i = 0; i < num_detections; ++i) {
        float* detection = output.ptr<float>(0, i);

        float obj_confidence = detection[4];
        if (obj_confidence < config_.confidence_threshold) continue;

        int class_id = 0;
        float max_class_score = detection[5];
        for (int c = 1; c < num_classes; ++c) {
            if (detection[5 + c] > max_class_score) {
                max_class_score = detection[5 + c];
                class_id = c;
            }
        }

        float final_confidence = obj_confidence * max_class_score;
        if (final_confidence < config_.confidence_threshold) continue;

        int center_x = static_cast<int>(detection[0]);
        int center_y = static_cast<int>(detection[1]);
        int box_width = static_cast<int>(detection[2]);
        int box_height = static_cast<int>(detection[3]);

        results.emplace_back(
            class_id,
            final_confidence,
            center_x - box_width / 2,
            center_y - box_height / 2,
            box_width,
            box_height
        );
    }
#endif

    return results;
}

bool OpenCvBackend::isSupported() const {
#if YOLO_OPENCV_BACKEND
    return true;
#else
    return false;
#endif
}

std::string OpenCvBackend::getName() const {
    return "OpenCV";
}

} // namespace yolo