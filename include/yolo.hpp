#pragma once

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

/**
 * @brief Detection result structure
 */
struct YOLO_API DetectResult {
    int class_id;
    float confidence;
    int x, y, width, height;

    DetectResult() : class_id(0), confidence(0.0f), x(0), y(0), width(0), height(0) {}
    DetectResult(int id, float conf, int px, int py, int w, int h)
        : class_id(id), confidence(conf), x(px), y(py), width(w), height(h) {}
};

/**
 * @brief Model configuration structure
 */
struct YOLO_API ModelConfig {
    std::string model_path;
    std::string label_path;
    int input_width;
    int input_height;
    float confidence_threshold;
    float nms_threshold;
    int num_threads;
    bool is_yolov8_or_later;

    ModelConfig()
        : input_width(640)
        , input_height(640)
        , confidence_threshold(0.5f)
        , nms_threshold(0.45f)
        , num_threads(4)
        , is_yolov8_or_later(true) {}
};

/**
 * @brief YOLO model version enumeration
 */
enum class YOLO_API ModelVersion {
    YOLO_V5,
    YOLO_V6,
    YOLO_V7,
    YOLO_V8,
    YOLO_V9,
    YOLO_V10,
    YOLO_V11,
    YOLO_NAS
};

/**
 * @brief Backend type enumeration
 */
enum class YOLO_API BackendType {
    CPU,
    OPENCV,
    OPENVINO,
    TENSORRT,
    COREML
};

/**
 * @brief Opaque image container for cross-platform support
 */
class YOLO_API ImageData {
public:
    virtual ~ImageData() = default;

protected:
    ImageData() = default;
    ImageData(const ImageData&) = default;
};

/**
 * @brief Create ImageData from raw pixels
 */
YOLO_API std::unique_ptr<ImageData> createImageData(
    const uint8_t* data,
    int width,
    int height,
    int channels,
    bool is_bgr
);

/**
 * @brief YOLO detector interface
 */
class YOLO_API Detector {
public:
    virtual ~Detector() = default;

    virtual bool initialize(const ModelConfig& config) = 0;
    virtual std::vector<DetectResult> detect(const ImageData& image) = 0;

    virtual void setConfidenceThreshold(float threshold) = 0;
    virtual void setNmsThreshold(float threshold) = 0;
    virtual bool isInitialized() const = 0;

    virtual BackendType getBackendType() const = 0;
    virtual ModelVersion getModelVersion() const = 0;
    virtual std::string getBackendName() const = 0;
    virtual std::string getVersionName() const = 0;
};

/**
 * @brief Create a detector instance
 */
YOLO_API std::unique_ptr<Detector> createDetector(
    ModelVersion version,
    BackendType backend
);

/**
 * @brief Get human-readable version name
 */
YOLO_API std::string getVersionName(ModelVersion version);

/**
 * @brief Get human-readable backend name
 */
YOLO_API std::string getBackendName(BackendType backend);

} // namespace yolo