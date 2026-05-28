#include <yolo.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>\n";
        std::cerr << "Note: This is a placeholder example. Actual inference\n";
        std::cerr << "      requires proper model file and image data.\n";
        return 1;
    }

    // Create detector with YOLOv8 and CPU backend
    auto detector = yolo::createDetector(
        yolo::ModelVersion::YOLO_V8,
        yolo::BackendType::CPU
    );

    if (!detector) {
        std::cerr << "Failed to create detector\n";
        return 1;
    }

    std::cout << "Created detector: " << detector->getVersionName()
              << " with " << detector->getBackendName() << " backend\n";

    // Configure detector
    yolo::ModelConfig config;
    config.model_path = argv[1];
    config.input_width = 640;
    config.input_height = 640;
    config.confidence_threshold = 0.5f;
    config.nms_threshold = 0.45f;
    config.is_yolov8_or_later = true;

    std::cout << "Model config: " << config.model_path << "\n";
    std::cout << "Input size: " << config.input_width << "x" << config.input_height << "\n";
    std::cout << "Confidence threshold: " << config.confidence_threshold << "\n";

    return 0;
}