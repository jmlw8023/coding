#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

struct BoundingBox {
    int x, y, width, height;
};

struct YoloResult {
    int class_id;
    float confidence;
    BoundingBox bbox;
    std::string class_name;
};

struct YoloConfig {
    std::string model_path;
    std::string label_path;
    std::string engine_path;

    int input_width = 640;
    int input_height = 640;

    float confidence_threshold = 0.5f;
    float nms_threshold = 0.45f;

    int num_threads = 4;

    bool is_yolov8_or_later = true;
};

enum class YoloVersion {
    YOLO_V5,
    YOLO_V6,
    YOLO_V7,
    YOLO_V8,
    YOLO_V9,
    YOLO_V10,
    YOLO_V11,
    YOLO_NAS
};

enum class BackendType {
    BACKEND_CPU,
    BACKEND_OPENCV,
    BACKEND_OPENVINO,
    BACKEND_TENSORRT,
    BACKEND_COREML
};

class YoloInterface {
public:
    virtual ~YoloInterface() = default;

    virtual bool initialize(const YoloConfig& config) = 0;
    virtual std::vector<YoloResult> detect(const uint8_t* data, int width, int height, int channels) = 0;
    virtual std::vector<YoloResult> detect(const cv::Mat& image) = 0;
    virtual void setConfidenceThreshold(float threshold) = 0;
    virtual void setNmsThreshold(float threshold) = 0;
    virtual bool isInitialized() const = 0;
    virtual BackendType getBackendType() const = 0;
    virtual YoloVersion getYoloVersion() const = 0;
};

class YoloFactory {
public:
    static std::unique_ptr<YoloInterface> create(YoloVersion version, BackendType backend);
    static std::string getBackendName(BackendType backend);
    static std::string getVersionName(YoloVersion version);
};