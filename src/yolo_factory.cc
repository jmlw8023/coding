/**
 * @file yolo_factory.cc
 * @brief YoloInterface factory implementation source file
 */

#include "detail/detector_impl.hpp"
#include "backend/ibackend.hpp"
#include <stdexcept>
#include <cstring>

namespace yolo {

// ============================================================
// ImageDataImpl Implementation
// ============================================================

ImageDataImpl::ImageDataImpl(const uint8_t* d, int w, int h, int c, bool bgr)
    : width(w), height(h), channels(c), is_bgr(bgr) {
    data.assign(d, d + w * h * c);
}

// ============================================================
// DetectorImpl Implementation
// ============================================================

DetectorImpl::DetectorImpl(ModelVersion version, std::unique_ptr<IBackend> backend)
    : version_(version)
    , backend_type_(BackendType::CPU)
    , backend_(std::move(backend))
{}

DetectorImpl::~DetectorImpl() = default;

bool DetectorImpl::initialize(const ModelConfig& config) {
    config_ = config;
    return backend_->initialize(config);
}

std::vector<DetectResult> DetectorImpl::detect(const ImageData& image) {
    if (!backend_ || !backend_->isSupported()) {
        return {};
    }

    auto* img_data = dynamic_cast<const ImageDataImpl*>(&image);
    if (!img_data) {
        return {};
    }

    return backend_->infer(
        img_data->data.data(),
        img_data->width,
        img_data->height,
        img_data->channels,
        img_data->is_bgr
    );
}

void DetectorImpl::setConfidenceThreshold(float threshold) {
    config_.confidence_threshold = threshold;
}

void DetectorImpl::setNmsThreshold(float threshold) {
    config_.nms_threshold = threshold;
}

bool DetectorImpl::isInitialized() const {
    return backend_ && backend_->isSupported();
}

BackendType DetectorImpl::getBackendType() const {
    return backend_type_;
}

ModelVersion DetectorImpl::getModelVersion() const {
    return version_;
}

std::string DetectorImpl::getBackendName() const {
    return backend_ ? backend_->getName() : "Unknown";
}

std::string DetectorImpl::getVersionName() const {
    return yolo::getVersionName(version_);
}

// ============================================================
// Factory Functions
// ============================================================

std::unique_ptr<Detector> createDetector(ModelVersion version, BackendType backend) {
    std::unique_ptr<IBackend> backend_impl;

    switch (backend) {
        case BackendType::CPU:
            backend_impl = std::make_unique<CpuBackend>();
            break;
#if YOLO_OPENCV_BACKEND
        case BackendType::OPENCV:
            backend_impl = std::make_unique<OpenCvBackend>();
            break;
#endif
#if YOLO_OPENVINO_BACKEND
        case BackendType::OPENVINO:
            backend_impl = std::make_unique<OpenVinoBackend>();
            break;
#endif
        default:
            return nullptr;
    }

    return std::make_unique<DetectorImpl>(version, std::move(backend_impl));
}

std::string getVersionName(ModelVersion version) {
    switch (version) {
        case ModelVersion::YOLO_V5: return "YOLOv5";
        case ModelVersion::YOLO_V6: return "YOLOv6";
        case ModelVersion::YOLO_V7: return "YOLOv7";
        case ModelVersion::YOLO_V8: return "YOLOv8";
        case ModelVersion::YOLO_V9: return "YOLOv9";
        case ModelVersion::YOLO_V10: return "YOLOv10";
        case ModelVersion::YOLO_V11: return "YOLOv11";
        case ModelVersion::YOLO_NAS: return "YOLO-NAS";
    }
    return "Unknown";
}

std::string getBackendName(BackendType backend) {
    switch (backend) {
        case BackendType::CPU: return "CPU";
        case BackendType::OPENCV: return "OpenCV";
        case BackendType::OPENVINO: return "OpenVINO";
        case BackendType::TENSORRT: return "TensorRT";
        case BackendType::COREML: return "CoreML";
    }
    return "Unknown";
}

std::unique_ptr<ImageData> createImageData(
    const uint8_t* data,
    int width,
    int height,
    int channels,
    bool is_bgr
) {
    return std::make_unique<ImageDataImpl>(data, width, height, channels, is_bgr);
}

} // namespace yolo