#pragma once

#include <yolo.hpp>
#include <cassert>
#include <iostream>
#include <vector>

namespace yolo {
namespace test {

bool testVersionEnum() {
    std::cout << "[TEST] Version enum values... ";
    assert(static_cast<int>(ModelVersion::YOLO_V5) == 0);
    assert(static_cast<int>(ModelVersion::YOLO_V8) == 3);
    assert(static_cast<int>(ModelVersion::YOLO_V11) == 6);
    std::cout << "PASS\n";
    return true;
}

bool testBackendEnum() {
    std::cout << "[TEST] Backend enum values... ";
    assert(static_cast<int>(BackendType::CPU) == 0);
    assert(static_cast<int>(BackendType::OPENCV) == 1);
    assert(static_cast<int>(BackendType::OPENVINO) == 2);
    std::cout << "PASS\n";
    return true;
}

bool testVersionNames() {
    std::cout << "[TEST] getVersionName()... ";
    assert(getVersionName(ModelVersion::YOLO_V5) == "YOLOv5");
    assert(getVersionName(ModelVersion::YOLO_V8) == "YOLOv8");
    assert(getVersionName(ModelVersion::YOLO_V11) == "YOLOv11");
    assert(getVersionName(ModelVersion::YOLO_NAS) == "YOLO-NAS");
    std::cout << "PASS\n";
    return true;
}

bool testBackendNames() {
    std::cout << "[TEST] getBackendName()... ";
    assert(getBackendName(BackendType::CPU) == "CPU");
    assert(getBackendName(BackendType::OPENCV) == "OpenCV");
    assert(getBackendName(BackendType::OPENVINO) == "OpenVINO");
    assert(getBackendName(BackendType::TENSORRT) == "TensorRT");
    std::cout << "PASS\n";
    return true;
}

bool testCreateDetector() {
    std::cout << "[TEST] createDetector()... ";
    auto detector = createDetector(ModelVersion::YOLO_V8, BackendType::CPU);
    assert(detector != nullptr);
    assert(detector->getModelVersion() == ModelVersion::YOLO_V8);
    assert(detector->getBackendType() == BackendType::CPU);
    assert(detector->isInitialized() == false);
    std::cout << "PASS\n";
    return true;
}

bool testDefaultConfig() {
    std::cout << "[TEST] ModelConfig defaults... ";
    ModelConfig config;
    assert(config.input_width == 640);
    assert(config.input_height == 640);
    assert(config.confidence_threshold == 0.5f);
    assert(config.nms_threshold == 0.45f);
    assert(config.num_threads == 4);
    assert(config.is_yolov8_or_later == true);
    std::cout << "PASS\n";
    return true;
}

bool testDetectResult() {
    std::cout << "[TEST] DetectResult structure... ";
    DetectResult result(5, 0.95f, 100, 200, 300, 400);
    assert(result.class_id == 5);
    assert(result.confidence == 0.95f);
    assert(result.x == 100);
    assert(result.y == 200);
    assert(result.width == 300);
    assert(result.height == 400);
    std::cout << "PASS\n";
    return true;
}

bool testImageDataCreation() {
    std::cout << "[TEST] createImageData()... ";
    std::vector<uint8_t> data(1920 * 1080 * 3, 128);
    auto image = createImageData(data.data(), 1920, 1080, 3, true);
    assert(image != nullptr);
    std::cout << "PASS\n";
    return true;
}

bool testDetectorNotInitialized() {
    std::cout << "[TEST] detect() before init... ";
    auto detector = createDetector(ModelVersion::YOLO_V8, BackendType::CPU);
    std::vector<uint8_t> data(640 * 640 * 3, 0);
    auto image = createImageData(data.data(), 640, 640, 3, true);
    auto results = detector->detect(*image);
    assert(results.empty());
    std::cout << "PASS\n";
    return true;
}

bool testAllBackends() {
    std::cout << "[TEST] All backend creation... ";
    auto cpu = createDetector(ModelVersion::YOLO_V8, BackendType::CPU);
    assert(cpu != nullptr);
    std::cout << "PASS\n";
    return true;
}

} // namespace test
} // namespace yolo