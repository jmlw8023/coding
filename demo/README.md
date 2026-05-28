# YoloInterface Demo

This directory contains a ready-to-use distribution package for other engineers.

## Directory Structure

```
demo/
├── include/
│   └── yolo.hpp       # Interface header file
├── lib/
│   ├── yolo_interface.lib   # Import library
│   └── yolo_interface.dll  # Dynamic library
├── README.md          # This file
├── CMakeLists.txt     # CMake configuration
└── main.cpp          # Example code
```

## Quick Start

### CMake Integration

```cmake
cmake_minimum_required(VERSION 3.15)
project(YoloDemo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_executable(yolo_demo main.cpp)

target_include_directories(yolo_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(yolo_demo PRIVATE
    ${OpenCV_LIBS}
    yolo_interface
)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
```

### Code Example

```cpp
#include <yolo.hpp>

int main() {
    // Create detector
    auto detector = yolo::createDetector(
        yolo::ModelVersion::YOLO_V8,
        yolo::BackendType::CPU
    );

    // Configure
    yolo::ModelConfig config;
    config.model_path = "yolov8n.onnx";
    config.confidence_threshold = 0.5f;

    // Initialize
    if (!detector->initialize(config)) {
        return -1;
    }

    // Prepare image data (BGR format)
    uint8_t* image_data = ...;
    int width = 1920;
    int height = 1080;
    auto image = yolo::createImageData(image_data, width, height, 3, true);

    // Detect
    auto results = detector->detect(*image);

    // Process results
    for (const auto& r : results) {
        printf("Class=%d Conf=%.2f Box=[%d,%d,%d,%d]\n",
               r.class_id, r.confidence, r.x, r.y, r.width, r.height);
    }

    return 0;
}
```

## Build and Run

### Windows (Visual Studio)

```batch
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Linux/macOS

```bash
mkdir build
cd build
cmake ..
make
```

## API Overview

### Create Detector

```cpp
std::unique_ptr<yolo::Detector> detector = yolo::createDetector(
    yolo::ModelVersion::YOLO_V8,   // YOLO_V5 ~ YOLO_V11, YOLO_NAS
    yolo::BackendType::CPU         // CPU, OPENCV, OPENVINO, TENSORRT, COREML
);
```

### Model Configuration

```cpp
yolo::ModelConfig config;
config.model_path = "model.onnx";       // Required
config.input_width = 640;               // Input width
config.input_height = 640;               // Input height
config.confidence_threshold = 0.5f;      // Confidence threshold
config.nms_threshold = 0.45f;           // NMS threshold
```

### Detection Result

```cpp
struct DetectResult {
    int class_id;      // Class ID
    float confidence;  // Confidence score
    int x, y;         // Top-left coordinates
    int width, height; // Bounding box size
};
```

## Runtime Dependencies

| Backend | Dependency |
|---------|------------|
| CPU | Standard library only |
| OPENCV | OpenCV 4.x |
| OPENVINO | OpenVINO Runtime |

## Notes

1. The DLL `yolo_interface.dll` must be accessible at runtime
2. OPENCV backend requires OpenCV to be installed
3. OPENVINO backend requires Intel OpenVINO Toolkit

## Technical Support

For issues, please submit an Issue or contact the maintainer.