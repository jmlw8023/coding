#ifndef YOLO_DETECTOR_IMPL_HPP
#define YOLO_DETECTOR_IMPL_HPP

/**
 * @file detector_impl.hpp
 * @brief Detector internal implementation header
 *
 * Defines internal implementation details including image data class and detector class.
 * These classes are not exposed externally, for library internal use only.
 */

#include "yolo.hpp"
#include <memory>

namespace yolo {

// Forward declaration
class IBackend;

// ============================================================
// ImageData Implementation Class
// ============================================================

/**
 * @brief Image data concrete implementation class
 *
 * Stores actual image pixel data, concrete implementation of ImageData.
 * Created via createImageData() factory function.
 *
 * @note Internal implementation class, not for user code
 */
class ImageDataImpl : public ImageData {
public:
    std::vector<uint8_t> data;  // Pixel data copy
    int width;                  // Image width
    int height;                 // Image height
    int channels;               // Channel count
    bool is_bgr;                // BGR format flag

    /** @brief Constructor */
    ImageDataImpl(const uint8_t* d, int w, int h, int c, bool bgr);
};

// ============================================================
// Detector Implementation Class
// ============================================================

/**
 * @brief Detector implementation class
 *
 * Actual detector implementation, encapsulating backend and configuration logic.
 * Created via createDetector() factory function.
 *
 * @note Internal implementation class, not for user code
 */
class DetectorImpl : public Detector {
public:
    /** @brief Constructor */
    DetectorImpl(ModelVersion version, std::unique_ptr<IBackend> backend);
    ~DetectorImpl() override;

    // Interface implementation
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
    ModelVersion version_;                    // YOLO model version
    BackendType backend_type_;               // Backend type
    ModelConfig config_;                      // Model configuration
    std::unique_ptr<IBackend> backend_;      // Backend smart pointer
};

} // namespace yolo

#endif // YOLO_DETECTOR_IMPL_HPP