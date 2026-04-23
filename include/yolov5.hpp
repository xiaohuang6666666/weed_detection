#pragma once

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>

// 检测框
struct Box {
    float x1, y1, x2, y2;    // 检测框坐标
    float score;
    int class_id;
};

// YOLOv5 推理类
class YOLOv5 {
public:
    // 构造：加载引擎
    YOLOv5(const std::string& enginePath);
    // 析构：释放资源
    ~YOLOv5();

    // 推理接口
    std::vector<Box> detect(const cv::Mat& img);


private:
    // 预处理,将输入图像转换为模型输入格式
    void preprocess(const cv::Mat& img, float* inputCPU);
    // 推理,将模型输入数据发送到GPU上执行推理
    void infer(float* inputCPU, float* outputCPU);
    // 后处理,将原始模型输出转换为可用的检测框坐标
    std::vector<Box> postprocess(const float* output, int imgW, int imgH);

    // TensorRT
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;            // CUDA 引擎
    nvinfer1::IExecutionContext* context = nullptr;     // 推理上下文
    cudaStream_t stream;            // CUDA 流（用于异步操作）

    float* d_input = nullptr;       // GPU上的输入缓冲区
    float* d_output = nullptr;      // GPU上的输出缓冲区
    float* d_out345  = nullptr;
    float* d_out473  = nullptr;
    float* d_out601  = nullptr;

    // 复用CPU缓冲区，减少每帧 new/delete 开销
    std::vector<float> inputCPUBuffer;
    std::vector<float> outputCPUBuffer;

    // 固定参数
    static const int INPUT_W = 640;
    static const int INPUT_H = 640;
    static const int NUM_BOXES = 25200;
    static const int NUM_ATTRS = 6;
};
