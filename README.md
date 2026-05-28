# YoloInterface

跨平台 YOLO 目标检测推理接口库

## 项目结构

```
yolo_interface/
├── include/
│   └── yolo.hpp               # 主接口头文件
├── src/
│   ├── yolo_factory.cc         # 工厂实现
│   └── backend/                # 后端实现
│       ├── cpu_backend.cc
│       ├── opencv_backend.cc
│       └── openvino_backend.cc
├── tests/                     # 测试代码
├── docs/                      # API 文档
└── demo/                       # 交付版本（可直接使用）
    ├── include/               # 接口头文件
    ├── lib/                   # 库文件
    ├── README.md              # 使用说明
    ├── CMakeLists.txt         # CMake 配置
    └── main.cpp               # 示例代码
```

## 快速开始

### 编译库

```bash
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### 使用 demo 目录（推荐）

`demo/` 目录包含完整交付资料，可直接提供给其他工程师：

```bash
cd demo
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## API 概览

```cpp
#include <yolo.hpp>

// 创建检测器
auto detector = yolo::createDetector(
    yolo::ModelVersion::YOLO_V8,
    yolo::BackendType::CPU
);

// 配置并初始化
yolo::ModelConfig config;
config.model_path = "yolov8n.onnx";
detector->initialize(config);

// 检测图像
auto image = yolo::createImageData(data, w, h, 3, true);
auto results = detector->detect(*image);
```

详细 API 说明请参考 [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

## 支持的后端

| 后端 | 说明 |
|------|------|
| CPU | 原生推理 |
| OPENCV | OpenCV DNN |
| OPENVINO | Intel OpenVINO |
| TENSORRT | NVIDIA TensorRT |
| COREML | Apple CoreML |

## 编译选项

| 选项 | 说明 | 默认 |
|------|------|------|
| `YOLO_WITH_OPENCV` | 启用 OpenCV 后端 | OFF |
| `YOLO_WITH_OPENVINO` | 启用 OpenVINO 后端 | OFF |