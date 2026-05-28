#pragma once

#include "yolo.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace yolo {

class IBackend {
public:
    virtual ~IBackend() = default;

    virtual bool initialize(const ModelConfig& config) = 0;
    virtual std::vector<DetectResult> infer(
        const uint8_t* data,
        int width,
        int height,
        int channels,
        bool is_bgr
    ) = 0;
    virtual bool isSupported() const = 0;
    virtual std::string getName() const = 0;

protected:
    ModelConfig config_;
    bool initialized_ = false;
};

class CpuBackend : public IBackend {
public:
    bool initialize(const ModelConfig& config) override;
    std::vector<DetectResult> infer(
        const uint8_t* data,
        int width,
        int height,
        int channels,
        bool is_bgr
    ) override;
    bool isSupported() const override;
    std::string getName() const override;

private:
    void preprocess(const uint8_t* input, int w, int h, int c, bool bgr, float* output);
    void postprocess(const float* output, int output_size, std::vector<DetectResult>& results);

    void* model_ptr_ = nullptr;
};

class OpenCvBackend : public IBackend {
public:
    bool initialize(const ModelConfig& config) override;
    std::vector<DetectResult> infer(
        const uint8_t* data,
        int width,
        int height,
        int channels,
        bool is_bgr
    ) override;
    bool isSupported() const override;
    std::string getName() const override;

private:
    void* net_ = nullptr;
    std::vector<std::string> output_names_;
};

class OpenVinoBackend : public IBackend {
public:
    bool initialize(const ModelConfig& config) override;
    std::vector<DetectResult> infer(
        const uint8_t* data,
        int width,
        int height,
        int channels,
        bool is_bgr
    ) override;
    bool isSupported() const override;
    std::string getName() const override;

private:
    void* core_ptr_ = nullptr;
    void* model_ptr_ = nullptr;
    void* compiled_model_ptr_ = nullptr;
    void* infer_request_ptr_ = nullptr;
};

} // namespace yolo