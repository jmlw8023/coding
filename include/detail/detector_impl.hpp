#ifndef YOLO_DETECTOR_IMPL_HPP
#define YOLO_DETECTOR_IMPL_HPP

#include "yolo.hpp"
#include <memory>

namespace yolo {

class IBackend;

class DetectorImpl : public Detector {
public:
    DetectorImpl(ModelVersion version, std::unique_ptr<IBackend> backend);
    ~DetectorImpl() override;

    bool initialize(const ModelConfig& config) override;
    std::vector<DetectResult> detect(const ImageData& image) override;

    void setConfidenceThreshold(float threshold) override;
    void setNmsThreshold(float threshold) override;
    bool isInitialized() const override;

    BackendType getBackendType() const override;
    ModelVersion getModelVersion() const override;
    std::string getBackendName() const override;
    std::string getVersionName() const override;

private:
    ModelVersion version_;
    BackendType backend_type_;
    ModelConfig config_;
    std::unique_ptr<IBackend> backend_;
};

class ImageDataImpl : public ImageData {
public:
    std::vector<uint8_t> data;
    int width;
    int height;
    int channels;
    bool is_bgr;

    ImageDataImpl(const uint8_t* d, int w, int h, int c, bool bgr);
};

} // namespace yolo

#endif // YOLO_DETECTOR_IMPL_HPP