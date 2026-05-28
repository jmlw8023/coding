#include <yolo.hpp>
#include <iostream>
#include <vector>
#include <cstring>

int main(int argc, char* argv[]) {
    std::cout << "==========================================\n";
    std::cout << "       YoloInterface Demo                 \n";
    std::cout << "==========================================\n\n";

    // Create detector with YOLOv8 and CPU backend
    auto detector = yolo::createDetector(
        yolo::ModelVersion::YOLO_V8,
        yolo::BackendType::CPU
    );

    if (!detector) {
        std::cerr << "Failed to create detector\n";
        return 1;
    }

    std::cout << "Detector created: " << detector->getVersionName()
              << " with " << detector->getBackendName() << " backend\n\n";

    // Show available options
    std::cout << "==========================================\n";
    std::cout << "Usage: This demo shows API usage patterns.\n";
    std::cout << "==========================================\n\n";

    // List all available versions
    std::cout << "Available YOLO versions:\n";
    std::cout << "  - YOLOv5: " << yolo::getVersionName(yolo::ModelVersion::YOLO_V5) << "\n";
    std::cout << "  - YOLOv6: " << yolo::getVersionName(yolo::ModelVersion::YOLO_V6) << "\n";
    std::cout << "  - YOLOv7: " << yolo::getVersionName(yolo::ModelVersion::YOLO_V7) << "\n";
    std::cout << "  - YOLOv8: " << yolo::getVersionName(yolo::ModelVersion::YOLO_V8) << "\n";
    std::cout << "  - YOLOv9: " << yolo::getVersionName(yolo::ModelVersion::YOLO_V9) << "\n";
    std::cout << "  - YOLOv10: " << yolo::getVersionName(yolo::ModelVersion::YOLO_V10) << "\n";
    std::cout << "  - YOLOv11: " << yolo::getVersionName(yolo::ModelVersion::YOLO_V11) << "\n";
    std::cout << "  - YOLO-NAS: " << yolo::getVersionName(yolo::ModelVersion::YOLO_NAS) << "\n\n";

    // List all available backends
    std::cout << "Available backends:\n";
    std::cout << "  - CPU: " << yolo::getBackendName(yolo::BackendType::CPU) << "\n";
    std::cout << "  - OPENCV: " << yolo::getBackendName(yolo::BackendType::OPENCV) << "\n";
    std::cout << "  - OPENVINO: " << yolo::getBackendName(yolo::BackendType::OPENVINO) << "\n";
    std::cout << "  - TENSORRT: " << yolo::getBackendName(yolo::BackendType::TENSORRT) << "\n";
    std::cout << "  - COREML: " << yolo::getBackendName(yolo::BackendType::COREML) << "\n\n";

    // Example: Create and configure detector
    std::cout << "==========================================\n";
    std::cout << "Example API usage:\n";
    std::cout << "==========================================\n";

    // Create another detector to show initialization pattern
    auto detector2 = yolo::createDetector(
        yolo::ModelVersion::YOLO_V8,
        yolo::BackendType::CPU
    );

    // Configure
    yolo::ModelConfig config;
    config.model_path = "yolov8n.onnx";
    config.input_width = 640;
    config.input_height = 640;
    config.confidence_threshold = 0.5f;
    config.nms_threshold = 0.45f;

    std::cout << "\nModel configuration:\n";
    std::cout << "  Model path: " << config.model_path << "\n";
    std::cout << "  Input size: " << config.input_width << "x" << config.input_height << "\n";
    std::cout << "  Confidence threshold: " << config.confidence_threshold << "\n";
    std::cout << "  NMS threshold: " << config.nms_threshold << "\n";

    // Create dummy image data
    std::vector<uint8_t> dummy_image(640 * 640 * 3, 128);
    auto image = yolo::createImageData(
        dummy_image.data(),
        640,
        640,
        3,
        true  // BGR format
    );

    std::cout << "\nImage data created: " << image->width << "x" << image->height
              << " channels=3 format=BGR\n";

    // Note: Without actual model file, we cannot complete initialization
    std::cout << "\nNote: To run actual inference, provide a valid ONNX model file.\n";
    std::cout << "The library is ready for integration.\n";

    std::cout << "\n==========================================\n";
    std::cout << "Demo completed successfully!\n";
    std::cout << "==========================================\n";

    return 0;
}