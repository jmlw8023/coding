#pragma once

/**
 * @file ibackend.hpp
 * @brief Backend abstract interface header
 *
 * Defines the abstract interface for all inference backends.
 * Provides extensible backend architecture.
 * Supports CPU, OpenCV, OpenVINO and more.
 */

#include "yolo.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace yolo {

// ============================================================
// Backend Abstract Interface
// ============================================================

/**
 * @brief Backend abstract interface class
 *
 * All inference backends must implement this interface.
 * Backend is responsible for actual model loading and inference execution.
 *
 * @note Internal interface, not for user code
 */
class IBackend {
public:
    virtual ~IBackend() = default;

    /** @brief Initialize backend */
    virtual bool initialize(const ModelConfig& config) = 0;

    /** @brief Execute inference */
    virtual std::vector<DetectResult> infer(
        const uint8_t* data,
        int width,
        int height,
        int channels,
        bool is_bgr
    ) = 0;

    /** @brief Check if backend is available */
    virtual bool isSupported() const = 0;

    /** @brief Get backend name */
    virtual std::string getName() const = 0;

protected:
    ModelConfig config_;
    bool initialized_ = false;
};

// ============================================================
// CPU Backend Implementation
// ============================================================

/**
 * @brief CPU backend implementation class
 *
 * Uses native CPU for inference, suitable for all platforms.
 * Most versatile backend, no external dependencies.
 *
 * @note Internal implementation class
 */
class CpuBackend : public IBackend {
public:
    bool initialize(const ModelConfig& config) override;
    std::vector<DetectResult> infer(
        const uint8_t* data,
        int width,
        int height,
        int channels,
        bool is_bgr
    ) override;
    bool isSupported() const override;
    std::string getName() const override;

private:
    void preprocess(const uint8_t* input, int w, int h, int c, bool bgr, float* output);
    void postprocess(const float* output, int output_size, std::vector<DetectResult>& results);

    void* model_ptr_ = nullptr;  // Model pointer (placeholder, actual uses ONNX Runtime)
};

// ============================================================
// OpenCV DNN Backend Implementation
// ============================================================

/**
 * @brief OpenCV DNN backend implementation class
 *
 * Uses OpenCV DNN module for inference.
 * Requires OpenCV 4.x or higher.
 *
 * @note Internal implementation class, requires YOLO_OPENCV_BACKEND compile flag
 */
class OpenCvBackend : public IBackend {
public:
    bool initialize(const ModelConfig& config) override;
    std::vector<DetectResult> infer(
        const uint8_t* data,
        int width,
        int height,
        int channels,
        bool is_bgr
    ) override;
    bool isSupported() const override;
    std::string getName() const override;

private:
    void* net_ = nullptr;
    std::vector<std::string> output_names_;
};

// ============================================================
// OpenVINO Backend Implementation
// ============================================================

/**
 * @brief OpenVINO backend implementation class
 *
 * Uses Intel OpenVINO Toolkit for inference.
 * Supports Intel CPU, GPU, NPU acceleration.
 *
 * @note Internal implementation class, requires YOLO_OPENVINO_BACKEND compile flag
 */
class OpenVinoBackend : public IBackend {
public:
    bool initialize(const ModelConfig& config) override;
    std::vector<DetectResult> infer(
        const uint8_t* data,
        int width,
        int height,
        int channels,
        bool is_bgr
    ) override;
    bool isSupported() const override;
    std::string getName() const override;

private:
    void* core_ptr_ = nullptr;
    void* model_ptr_ = nullptr;
    void* compiled_model_ptr_ = nullptr;
    void* infer_request_ptr_ = nullptr;
};

} // namespace yolo