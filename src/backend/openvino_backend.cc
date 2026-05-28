/**
 * @file openvino_backend.cc
 * @brief OpenVINO backend implementation source file
 *
 * Uses Intel OpenVINO Toolkit for YOLO inference.
 * Supports Intel CPU, GPU, NPU acceleration.
 */

#include "backend/ibackend.hpp"
#include <vector>

#if YOLO_OPENVINO_BACKEND
#include <openvino/openvino.hpp>
#endif

#if YOLO_OPENCV_BACKEND
#include <opencv2/opencv.hpp>
#endif

namespace yolo {

// ============================================================
// OpenVinoBackend Implementation
// ============================================================

bool OpenVinoBackend::initialize(const ModelConfig& config) {
    config_ = config;

#if YOLO_OPENVINO_BACKEND && YOLO_OPENCV_BACKEND
    try {
        ov::Core core;
        model_ptr_ = new ov::Model(core.read_model(config_.model_path));

        ov::CompiledModel compiled_model = core.compile_model(
            static_cast<ov::Model*>(model_ptr_),
            "CPU"
        );
        compiled_model_ptr_ = new ov::CompiledModel(compiled_model);

        ov::InferRequest infer_request = compiled_model.create_infer_request();
        infer_request_ptr_ = new ov::InferRequest(infer_request);

        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        return false;
    }
#else
    return false;
#endif
}

std::vector<DetectResult> OpenVinoBackend::infer(
    const uint8_t* data,
    int width,
    int height,
    int channels,
    bool is_bgr
) {
    std::vector<DetectResult> results;

#if YOLO_OPENVINO_BACKEND && YOLO_OPENCV_BACKEND
    if (!infer_request_ptr_) return results;

    cv::Mat image(height, width, channels == 3 ? CV_8UC3 : CV_8UC1, (void*)data);
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(config_.input_width, config_.input_height));
    cv::Mat blob;
    resized.convertTo(blob, CV_32FC3, 1.0 / 255.0);

    ov::InferRequest* req = static_cast<ov::InferRequest*>(infer_request_ptr_);
    auto input_port = req->get_input_port(0);
    input_port.set_tensor(ov::make_tensor(blob));
    req->infer();

    auto output_port = req->get_output_port(0);
    ov::Tensor output_tensor = output_port.get_tensor();

    const float* output_data = output_tensor.data<float>();
    size_t output_size = output_tensor.get_size();

    postprocess(output_data, static_cast<int>(output_size), results);
#endif

    return results;
}

bool OpenVinoBackend::isSupported() const {
#if YOLO_OPENVINO_BACKEND
    return true;
#else
    return false;
#endif
}

std::string OpenVinoBackend::getName() const {
    return "OpenVINO";
}

} // namespace yolo