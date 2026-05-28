/**
 * @file cpu_backend.cc
 * @brief CPU backend implementation source file
 *
 * Uses native CPU for YOLO inference.
 * This is the most versatile implementation with minimal dependencies.
 */

#include "backend/ibackend.hpp"
#include <cstring>

namespace yolo {

// ============================================================
// CpuBackend Implementation
// ============================================================

bool CpuBackend::initialize(const ModelConfig& config) {
    config_ = config;
    model_ptr_ = nullptr;  // Placeholder: actual should load ONNX model
    initialized_ = true;
    return true;
}

std::vector<DetectResult> CpuBackend::infer(
    const uint8_t* data,
    int width,
    int height,
    int channels,
    bool is_bgr
) {
    std::vector<DetectResult> results;

    // Preprocess: resize and normalize
    int target_w = config_.input_width;
    int target_h = config_.input_height;
    float* input_buffer = new float[target_w * target_h * 3];
    preprocess(data, width, height, channels, is_bgr, input_buffer);

    // Placeholder: actual inference should happen here
    // ONNX Runtime inference call goes here

    delete[] input_buffer;
    return results;
}

bool CpuBackend::isSupported() const {
    return true;
}

std::string CpuBackend::getName() const {
    return "CPU";
}

// ============================================================
// Preprocess Implementation
// ============================================================

void CpuBackend::preprocess(
    const uint8_t* input,
    int w,
    int h,
    int c,
    bool bgr,
    float* output
) {
    int target_w = config_.input_width;
    int target_h = config_.input_height;
    float scale_x = static_cast<float>(w) / target_w;
    float scale_y = static_cast<float>(h) / target_h;

    for (int y = 0; y < target_h; ++y) {
        for (int x = 0; x < target_w; ++x) {
            int src_x = static_cast<int>(x * scale_x);
            int src_y = static_cast<int>(y * scale_y);
            src_x = std::min(src_x, w - 1);
            src_y = std::min(src_y, h - 1);

            int src_idx = (src_y * w + src_x) * c;
            int dst_idx = (y * target_w + x) * 3;

            if (bgr && c == 3) {
                output[dst_idx + 0] = input[src_idx + 2] / 255.0f;  // R
                output[dst_idx + 1] = input[src_idx + 1] / 255.0f;  // G
                output[dst_idx + 2] = input[src_idx + 0] / 255.0f;  // B
            } else {
                for (int ch = 0; ch < 3; ++ch) {
                    output[dst_idx + ch] = input[src_idx + ch] / 255.0f;
                }
            }
        }
    }
}

void CpuBackend::postprocess(
    const float* output,
    int output_size,
    std::vector<DetectResult>& results
) {
    // Placeholder implementation
    // Parse YOLO output format based on version:
    // - YOLOv5/v7: [batch, num_detections, 5 + num_classes]
    // - YOLOv8+: uses new output format
}

} // namespace yolo