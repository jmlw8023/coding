# YoloInterface API Reference

跨平台 YOLO 目标检测推理接口库

## 概述

YoloInterface 是一个跨平台的 YOLO 目标检测推理库，提供统一的 C++ 接口，支持多种推理后端。

---

## 头文件

```cpp
#include <yolo.hpp>
```

---

## 核心类型

### DetectResult

检测结果结构体

```cpp
struct DetectResult {
    int class_id;       // 类别 ID (COCO 数据集: 0-79)
    float confidence;  // 置信度 [0.0, 1.0]
    int x;             // 左上角 X 坐标
    int y;             // 左上角 Y 坐标
    int width;         // 边界框宽度
    int height;        // 边界框高度
};
```

### ModelConfig

模型配置结构体

```cpp
struct ModelConfig {
    std::string model_path;           // ONNX 模型文件路径 (必需)
    std::string label_path;           // 类别标签文件路径 (可选)
    int input_width = 640;           // 模型输入宽度
    int input_height = 640;           // 模型输入高度
    float confidence_threshold = 0.5f; // 置信度阈值
    float nms_threshold = 0.45f;     // NMS 阈值
    int num_threads = 4;             // CPU 线程数
    bool is_yolov8_or_later = true;  // 是否为 YOLOv8 及之后版本
};
```

### ModelVersion

YOLO 模型版本枚举

```cpp
enum class ModelVersion {
    YOLO_V5,   // YOLOv5
    YOLO_V6,   // YOLOv6
    YOLO_V7,   // YOLOv7
    YOLO_V8,   // YOLOv8
    YOLO_V9,   // YOLOv9
    YOLO_V10,  // YOLOv10
    YOLO_V11,  // YOLOv11
    YOLO_NAS   // YOLO-NAS
};
```

### BackendType

推理后端类型枚举

```cpp
enum class BackendType {
    CPU,       // 原生 CPU 推理
    OPENCV,    // OpenCV DNN 后端
    OPENVINO,  // Intel OpenVINO (GPU/NPU 加速)
    TENSORRT,  // NVIDIA TensorRT
    COREML     // Apple CoreML
};
```

### ImageData

图像数据容器 (抽象基类)

```cpp
class ImageData {
public:
    virtual ~ImageData() = default;
};
```

创建 ImageData:

```cpp
std::unique_ptr<ImageData> createImageData(
    const uint8_t* data,  // 像素数据
    int width,           // 图像宽度
    int height,          // 图像高度
    int channels,        // 通道数 (3=彩色, 1=灰度)
    bool is_bgr          // 数据格式: true=BGR, false=RGB
);
```

### Detector

检测器接口类

```cpp
class Detector {
public:
    virtual ~Detector() = default;

    // 初始化检测器
    virtual bool initialize(const ModelConfig& config) = 0;

    // 检测图像
    virtual std::vector<DetectResult> detect(const ImageData& image) = 0;

    // 设置置信度阈值
    virtual void setConfidenceThreshold(float threshold) = 0;

    // 设置 NMS 阈值
    virtual void setNmsThreshold(float threshold) = 0;

    // 检查是否已初始化
    virtual bool isInitialized() const = 0;

    // 获取后端类型
    virtual BackendType getBackendType() const = 0;

    // 获取模型版本
    virtual ModelVersion getModelVersion() const = 0;

    // 获取后端名称
    virtual std::string getBackendName() const = 0;

    // 获取模型版本名称
    virtual std::string getVersionName() const = 0;
};
```

---

## 工厂函数

### createDetector

创建检测器实例

```cpp
std::unique_ptr<Detector> yolo::createDetector(
    ModelVersion version,   // YOLO 版本
    BackendType backend    // 推理后端
);
```

**示例:**

```cpp
auto detector = yolo::createDetector(
    yolo::ModelVersion::YOLO_V8,
    yolo::BackendType::CPU
);
```

### createImageData

创建图像数据容器

```cpp
std::unique_ptr<ImageData> yolo::createImageData(
    const uint8_t* data,
    int width,
    int height,
    int channels,
    bool is_bgr
);
```

### getVersionName

获取 YOLO 版本的人类可读名称

```cpp
std::string yolo::getVersionName(ModelVersion version);
// 返回: "YOLOv5", "YOLOv8", 等
```

### getBackendName

获取后端的人类可读名称

```cpp
std::string yolo::getBackendName(BackendType backend);
// 返回: "CPU", "OpenCV", "OpenVINO", 等
```

---

## 使用示例

### 完整示例

```cpp
#include <yolo.hpp>

int main() {
    // 1. 创建检测器 (YOLOv8 + CPU)
    auto detector = yolo::createDetector(
        yolo::ModelVersion::YOLO_V8,
        yolo::BackendType::CPU
    );

    if (!detector) {
        return -1;
    }

    // 2. 配置模型
    yolo::ModelConfig config;
    config.model_path = "yolov8n.onnx";
    config.input_width = 640;
    config.input_height = 640;
    config.confidence_threshold = 0.5f;
    config.nms_threshold = 0.45f;

    // 3. 初始化
    if (!detector->initialize(config)) {
        return -1;
    }

    // 4. 准备图像数据
    // 假设 image_data 是 BGR 格式的像素数据
    uint8_t* image_data = ...;
    int width = 1920;
    int height = 1080;
    auto image = yolo::createImageData(image_data, width, height, 3, true);

    // 5. 检测
    auto results = detector->detect(*image);

    // 6. 处理结果
    printf("Detected %zu objects\n", results.size());
    for (const auto& r : results) {
        printf("Class=%d Conf=%.2f Box=[%d,%d,%d,%d]\n",
               r.class_id, r.confidence,
               r.x, r.y, r.width, r.height);
    }

    return 0;
}
```

---

## 编译与使用

### CMake 依赖

```cmake
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_executable(your_app main.cpp)
target_link_libraries(your_app PRIVATE
    ${OpenCV_LIBS}
    yolo_interface
)
```

### 运行时依赖

| 后端 | 依赖 |
|------|------|
| CPU | 仅标准库 |
| OPENCV | OpenCV 4.x |
| OPENVINO | OpenVINO Runtime |
| TENSORRT | TensorRT |
| COREML | CoreML.framework |

### 编译选项

| CMake 选项 | 说明 | 默认 |
|------------|------|------|
| `YOLO_WITH_OPENCV` | 启用 OpenCV 后端 | OFF |
| `YOLO_WITH_OPENVINO` | 启用 OpenVINO 后端 | OFF |
| `YOLO_WITH_TENSORRT` | 启用 TensorRT 后端 | OFF |

---

## 错误处理

所有可能失败的操作都返回 `bool` 或 `nullptr`，请检查返回值:

```cpp
auto detector = yolo::createDetector(...);
if (!detector) {
    // 处理创建失败
}

if (!detector->initialize(config)) {
    // 处理初始化失败
}
```

---

## 线程安全性

- 每个 `Detector` 实例应在单一线程中使用
- 如需多线程，请创建多个 `Detector` 实例
- `ModelConfig` 和 `DetectResult` 为 POD 类型，可安全共享

---

## 平台支持

| 平台 | CPU | OpenCV | OpenVINO | TensorRT | CoreML |
|------|-----|--------|----------|----------|--------|
| Windows x64 | ✓ | ✓ | ✓ | - | - |
| Linux x64 | ✓ | ✓ | ✓ | ✓ | - |
| macOS x64 | ✓ | ✓ | - | - | ✓ |
| macOS ARM64 | ✓ | ✓ | - | - | ✓ |
| iOS | - | - | - | - | ✓ |

---

## 版本历史

| 版本 | 说明 |
|------|------|
| 1.0.0 | 初始版本，支持 CPU/OpenCV/OpenVINO 后端，跨平台设计 |