#pragma once

/**
 * @file yolo.hpp
 * @brief YoloInterface main header file
 *
 * This file defines the unified interface for YOLO object detection inference.
 * Supports multiple YOLO versions and inference backends.
 * Uses the yolo namespace for cross-platform C++ interface.
 */

#include <memory>
#include <vector>
#include <string>
#include <cstdint>

#if defined(_WIN32)
    #if defined(YOLO_EXPORTS)
        #define YOLO_API __declspec(dllexport)
    #elif defined(YOLO_IMPORTS)
        #define YOLO_API __declspec(dllimport)
    #else
        #define YOLO_API
    #endif
#else
    #define YOLO_API
#endif

namespace yolo {

// Forward declarations
class Detector;
class IBackend;

// ============================================================
// Enum Types
// ============================================================

/**
 * @brief YOLO model version enum
 *
 * Supported YOLO versions:
 * - YOLO_V5 ~ YOLO_V11: Ultralytics YOLO series
 * - YOLO_NAS: YOLO-NAS auto-generated models
 */
enum class YOLO_API ModelVersion {
    YOLO_V5 = 0,
    YOLO_V6 = 1,
    YOLO_V7 = 2,
    YOLO_V8 = 3,
    YOLO_V9 = 4,
    YOLO_V10 = 5,
    YOLO_V11 = 6,
    YOLO_NAS = 7
};

/**
 * @brief Inference backend type enum
 *
 * Different backends for various hardware:
 * - CPU: Native CPU inference
 * - OPENCV: OpenCV DNN module
 * - OPENVINO: Intel OpenVINO for GPU/NPU acceleration
 * - TENSORRT: NVIDIA TensorRT for GPU acceleration
 * - COREML: Apple CoreML for Apple Silicon
 */
enum class YOLO_API BackendType {
    CPU = 0,
    OPENCV = 1,
    OPENVINO = 2,
    TENSORRT = 3,
    COREML = 4
};

// ============================================================
// Struct Definitions
// ============================================================

/**
 * @brief Detection result structure
 *
 * Stores single detection result including class ID, confidence and bbox.
 * Coordinates are absolute pixel values based on input image.
 */
struct YOLO_API DetectResult {
    int class_id;       // Class ID (COCO: 0-79)
    float confidence;  // Confidence [0.0, 1.0]
    int x;              // Bbox top-left X coordinate
    int y;              // Bbox top-left Y coordinate
    int width;          // Bbox width
    int height;         // Bbox height

    /** @brief Default constructor */
    DetectResult()
        : class_id(0), confidence(0.0f), x(0), y(0), width(0), height(0) {}

    /** @brief Parameterized constructor */
    DetectResult(int id, float conf, int px, int py, int w, int h)
        : class_id(id), confidence(conf), x(px), y(py), width(w), height(h) {}
};

/**
 * @brief Model configuration structure
 *
 * Used to configure detector parameters. Must be properly set before calling initialize().
 * Note: model_path is required, other parameters have default values.
 */
struct YOLO_API ModelConfig {
    std::string model_path;              // ONNX model file path (required)
    std::string label_path;              // Class label file path (optional)
    int input_width = 640;               // Model input width
    int input_height = 640;              // Model input height
    float confidence_threshold = 0.5f;   // Confidence threshold
    float nms_threshold = 0.45f;         // NMS (Non-Maximum Suppression) threshold
    int num_threads = 4;                 // CPU inference thread count
    bool is_yolov8_or_later = true;       // YOLOv8 or later version flag

    /** @brief Default constructor with default values */
    ModelConfig()
        : input_width(640)
        , input_height(640)
        , confidence_threshold(0.5f)
        , nms_threshold(0.45f)
        , num_threads(4)
        , is_yolov8_or_later(true) {}
};

// ============================================================
// Image Data Container
// ============================================================

/**
 * @brief Image data container abstract base class
 *
 * Provides cross-platform image data encapsulation.
 * Use createImageData() to create instances.
 */
class YOLO_API ImageData {
public:
    virtual ~ImageData() = default;

protected:
    ImageData() = default;
    ImageData(const ImageData&) = default;
};

// ============================================================
// Detector Interface
// ============================================================

/**
 * @brief YOLO detector interface class
 *
 * Core interface providing all object detection functionality.
 * Create instances via createDetector() factory function.
 *
 * Note: All member functions are not thread-safe unless specified.
 */
class YOLO_API Detector {
public:
    virtual ~Detector() = default;

    /**
     * @brief Initialize detector
     * @param config Model configuration
     * @return true on success, false on failure
     * @note Must be called before detect()
     */
    virtual bool initialize(const ModelConfig& config) = 0;

    /**
     * @brief Detect objects in image
     * @param image Image data reference
     * @return Detection results vector
     */
    virtual std::vector<DetectResult> detect(const ImageData& image) = 0;

    /**
     * @brief Set confidence threshold
     * @param threshold New threshold [0.0, 1.0]
     */
    virtual void setConfidenceThreshold(float threshold) = 0;

    /**
     * @brief Set NMS threshold
     * @param threshold New threshold [0.0, 1.0]
     */
    virtual void setNmsThreshold(float threshold) = 0;

    /**
     * @brief Check if initialized
     * @return true if initialized
     */
    virtual bool isInitialized() const = 0;

    /**
     * @brief Get backend type
     * @return Current backend type
     */
    virtual BackendType getBackendType() const = 0;

    /**
     * @brief Get model version
     * @return Current model version
     */
    virtual ModelVersion getModelVersion() const = 0;

    /**
     * @brief Get backend name
     * @return Backend type as readable string
     */
    virtual std::string getBackendName() const = 0;

    /**
     * @brief Get model version name
     * @return Model version as readable string
     */
    virtual std::string getVersionName() const = 0;
};

// ============================================================
// Factory Functions
// ============================================================

/**
 * @brief Create detector instance
 *
 * Creates appropriate detector based on YOLO version and backend type.
 *
 * @param version YOLO model version
 * @param backend Inference backend type
 * @return Smart pointer to new detector, nullptr on failure
 */
YOLO_API std::unique_ptr<Detector> createDetector(
    ModelVersion version,
    BackendType backend
);

/**
 * @brief Create image data from raw pixels
 *
 * @param data Pixel data pointer (RGB or BGR format)
 * @param width Image width
 * @param height Image height
 * @param channels Channel count (3=color, 1=grayscale)
 * @param is_bgr Data format: true=BGR, false=RGB
 * @return Smart pointer to new image data
 */
YOLO_API std::unique_ptr<ImageData> createImageData(
    const uint8_t* data,
    int width,
    int height,
    int channels,
    bool is_bgr
);

/**
 * @brief Get human-readable YOLO version name
 * @param version YOLO version enum value
 * @return Version string like "YOLOv8"
 */
YOLO_API std::string getVersionName(ModelVersion version);

/**
 * @brief Get human-readable backend name
 * @param backend Backend type enum value
 * @return Backend name string like "CPU", "OpenCV"
 */
YOLO_API std::string getBackendName(BackendType backend);

} // namespace yolo