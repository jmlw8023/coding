# YoloInterface Demo

本目录为 YoloInterface 的交付版本，可直接提供给其他工程师使用。

## 目录结构

```
demo/
├── include/
│   └── yolo.hpp       # 接口头文件
├── lib/
│   ├── yolo_interface.lib   # 导入库
│   └── yolo_interface.dll  # 动态库
├── README.md          # 使用说明
├── CMakeLists.txt     # CMake 配置
└── main.cpp           # 示例代码
```

## 快速开始

### 1. CMake 集成

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

# 设置库路径
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
```

### 2. 代码示例

```cpp
#include <yolo.hpp>

int main() {
    // 创建检测器
    auto detector = yolo::createDetector(
        yolo::ModelVersion::YOLO_V8,
        yolo::BackendType::CPU
    );

    // 配置
    yolo::ModelConfig config;
    config.model_path = "yolov8n.onnx";
    config.confidence_threshold = 0.5f;

    // 初始化
    if (!detector->initialize(config)) {
        return -1;
    }

    // 准备图像数据 (BGR 格式)
    uint8_t* image_data = ...;
    int width = 1920;
    int height = 1080;
    auto image = yolo::createImageData(image_data, width, height, 3, true);

    // 检测
    auto results = detector->detect(*image);

    // 处理结果
    for (const auto& r : results) {
        printf("Class=%d Conf=%.2f Box=[%d,%d,%d,%d]\n",
               r.class_id, r.confidence, r.x, r.y, r.width, r.height);
    }

    return 0;
}
```

## 编译运行

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

## 接口说明

### 创建检测器

```cpp
std::unique_ptr<yolo::Detector> detector = yolo::createDetector(
    yolo::ModelVersion::YOLO_V8,   // YOLO_V5 ~ YOLO_V11, YOLO_NAS
    yolo::BackendType::CPU         // CPU, OPENCV, OPENVINO, TENSORRT, COREML
);
```

### 模型配置

```cpp
yolo::ModelConfig config;
config.model_path = "model.onnx";       // 模型路径 (必需)
config.input_width = 640;               // 输入宽度
config.input_height = 640;               // 输入高度
config.confidence_threshold = 0.5f;      // 置信度阈值
config.nms_threshold = 0.45f;           // NMS 阈值
```

### 检测结果

```cpp
struct DetectResult {
    int class_id;      // 类别 ID
    float confidence;  // 置信度
    int x, y;          // 左上角坐标
    int width, height; // 宽高
};
```

## 运行时依赖

| 后端 | 依赖 |
|------|------|
| CPU | 仅标准库 |
| OPENCV | OpenCV 4.x |
| OPENVINO | OpenVINO Runtime |

## 注意事项

1. 动态库 `yolo_interface.dll` 需要在运行时可访问
2. 使用 OPENCV 后端需要安装 OpenCV
3. 使用 OPENVINO 后端需要安装 Intel OpenVINO Toolkit

## 技术支持

如有问题请提交 Issue 或联系 maintainer。