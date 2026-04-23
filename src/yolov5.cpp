#include "yolov5.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

// 日志类（加载引擎用）
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// Step1：构造函数 → 加载引擎 + 分配显存
YOLOv5::YOLOv5(const std::string& enginePath) {
    // 1. 读取引擎文件
    std::ifstream f(enginePath, std::ios::binary | std::ios::ate);
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<char> data(size);
    f.read(data.data(), size);

    // 2. 加载 TensorRT 引擎
    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(data.data(), size);
    context = engine->createExecutionContext();

    // 3. 创建 CUDA 流
    cudaStreamCreate(&stream);
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount <= 0) 
    {
        throw std::runtime_error("未检测到可用 CUDA 设备，无法使用 GPU 推理");
    }
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::cout << "[CUDA] device=" << prop.name
              << ", capability=" << prop.major << "." << prop.minor
              << ", globalMemMB=" << (prop.totalGlobalMem / (1024 * 1024))
              << std::endl;

    // 4. 分配 GPU 显存
    cudaMalloc(&d_input,  1 * 3 * 640 * 640 * sizeof(float));
    cudaMalloc(&d_output, 1 * 25200 * 6 * sizeof(float));
    cudaMalloc(&d_out345, 1 * 3 * 80 * 80 * 6 * sizeof(float));
    cudaMalloc(&d_out473, 1 * 3 * 40 * 40 * 6 * sizeof(float));
    cudaMalloc(&d_out601, 1 * 3 * 20 * 20 * sizeof(float) * 6);

    // 5. 绑定 tensor
    context->setTensorAddress("images", d_input);
    context->setTensorAddress("output", d_output);

    // 6. 预分配 CPU 缓冲区（避免每帧 new/delete 开销）
    inputCPUBuffer.resize(1 * 3 * INPUT_H * INPUT_W);
    outputCPUBuffer.resize(1 * NUM_BOXES * NUM_ATTRS);

    std::cout << "✅ YOLOv5 初始化成功！" << std::endl;
}

YOLOv5::~YOLOv5() {
    // 释放显存
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_out345);
    cudaFree(d_out473);
    cudaFree(d_out601);
    // 释放流
    cudaStreamDestroy(stream);

    // 释放TensorRT
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;

    std::cout << "✅ 资源已释放" << std::endl;
}

std::vector<Box> YOLOv5::detect(const cv::Mat& img) 
{
    // 1. 预处理
    preprocess(img, inputCPUBuffer.data());

    // 2. 推理
    infer(inputCPUBuffer.data(), outputCPUBuffer.data());

    // 3. 后处理
    std::vector<Box> boxes = postprocess(outputCPUBuffer.data(), img.cols, img.rows);
    return boxes;
}

void YOLOv5::preprocess(const cv::Mat& img, float* inputCPU) 
{
    // 1、缩放到640x640
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
    // 2、BGR → RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    // 3、归一化 0~1
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    // 4、HWC → CHW
    for (int h = 0; h < INPUT_H; h++) {
        for (int w = 0; w < INPUT_W; w++) {
            cv::Vec3f pix = resized.at<cv::Vec3f>(h, w);
            inputCPU[0 * INPUT_H * INPUT_W + h * INPUT_W + w] = pix[0];
            inputCPU[1 * INPUT_H * INPUT_W + h * INPUT_W + w] = pix[1];
            inputCPU[2 * INPUT_H * INPUT_W + h * INPUT_W + w] = pix[2];
        }
    }
}

void YOLOv5::infer(float* inputCPU, float* outputCPU)
{
    // 1、将输入数据从 CPU 复制到 GPU(异步非阻塞模式)
    cudaMemcpyAsync(d_input, inputCPU, 1 * 3 * 640 * 640 * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 2、执行推理（tensorrt10.7 推理接口）告诉 TensorRT 推理时去哪里读取输入、哪里写入输出数据
    context->setTensorAddress("images", d_input);    // 输入张量名
    context->setTensorAddress("output", d_output);   // 输出张量名
    context->setTensorAddress("345", d_out345);   
    context->setTensorAddress("473", d_out473);   
    context->setTensorAddress("601", d_out601); 
    context->enqueueV3(stream);                      // 执行推理     
    cudaStreamSynchronize(stream);  // 等待推理完成（阻塞 CPU，直到 GPU 上的所有操作完成）

    // 3、将输出数据从 GPU 复制到 CPU
    cudaMemcpyAsync(outputCPU, d_output, 1 * 25200 * 6 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

std::vector<Box> YOLOv5::postprocess(const float* outputCPU, int imgW, int imgH) 
{
    std::vector<Box> boxes;
    // 阈值
    const float confThreshold = 0.25f;  // 置信度阈值
    const float objThreshold  = 0.25f;  // 物体ness阈值
    const float nmsThreshold  = 0.45f;  // NMS阈值

    // YOLOv5 预处理的 letterbox 会有 padding
    // 这里按等比例缩放计算 padding 和缩放比例，和预处理保持一致
    float scale = std::min((float)INPUT_W / imgW, (float)INPUT_H / imgH);   
    float padw = (INPUT_W - imgW * scale) / 2.0f;    // 宽度 padding
    float padh = (INPUT_H - imgH * scale) / 2.0f;    // 高度 padding
    float ratiow = 1.0f / scale;     // 宽度缩放比例    
    float ratioh = 1.0f / scale;     // 高度缩放比例
    
    for (int i = 0; i < NUM_BOXES; i++) 
    {
        // 输出格式：x, y, w, h, objectness, class_score
        float x = outputCPU[i * 6 + 0];    // 中心x坐标
        float y = outputCPU[i * 6 + 1];    // 中心y坐标
        float w = outputCPU[i * 6 + 2];    // 宽度
        float h = outputCPU[i * 6 + 3];    // 高度
        float obj = outputCPU[i * 6 + 4];    // 物体ness
        float cls = outputCPU[i * 6 + 5];    // 类别分数
        float confidence = obj * cls;

        // 双重阈值过滤（和参考代码一致）
        if (confidence < confThreshold || obj < objThreshold)
            continue;

        // 转换成左上角+右下角坐标
        float xmin = x - w / 2.0f;
        float ymin = y - h / 2.0f;
        float xmax = x + w / 2.0f;
        float ymax = y + h / 2.0f;

        // 还原 padding + 缩放 目前还是640x640的坐标，需要转换回原图坐标
        xmin = (xmin - padw) * ratiow;
        ymin = (ymin - padh) * ratioh;
        xmax = (xmax - padw) * ratiow;
        ymax = (ymax - padh) * ratioh;

        // 坐标裁剪（防止越界崩溃）imgW - 1是因为坐标从0开始，而图片从1开始，所以要减1
        xmin = std::max(0.0f, std::min(xmin, (float)imgW - 1));
        ymin = std::max(0.0f, std::min(ymin, (float)imgH - 1));
        xmax = std::max(0.0f, std::min(xmax, (float)imgW - 1));
        ymax = std::max(0.0f, std::min(ymax, (float)imgH - 1));

        // 只保留有效框
        if (xmax > xmin && ymax > ymin) {
            boxes.push_back({xmin, ymin, xmax, ymax, confidence, 0});
        }
    }

    // NMS 非极大值抑制去重框
    // 排序的目的是让高置信度的框优先保留，低置信度的框更容易被抑制掉
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.score > b.score;
    });
    std::vector<bool> remove_flag(boxes.size(), false);
    for (int i = 0; i < boxes.size(); i++) 
    {
        if (remove_flag[i]) continue;
        Box& a = boxes[i];
        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        for (int j = i + 1; j < boxes.size(); j++) 
        {
            if (remove_flag[j]) continue;
            Box& b = boxes[j];
            float inter_x1 = std::max(a.x1, b.x1);
            float inter_y1 = std::max(a.y1, b.y1);
            float inter_x2 = std::min(a.x2, b.x2);
            float inter_y2 = std::min(a.y2, b.y2);

            float w = std::max(0.0f, inter_x2 - inter_x1);
            float h = std::max(0.0f, inter_y2 - inter_y1);
            float inter = w * h;     // 交集面积

            float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
            float iou = inter / (area_a + area_b - inter + 1e-6);

            if (iou > nmsThreshold) 
            {
                remove_flag[j] = true;  //对i而言，j是低置信且重叠过大的框，标记为删除
            }
        }
    }

    // 最终干净的框
    std::vector<Box> final_boxes;
    for (int i = 0; i < boxes.size(); i++) 
    {
        if (!remove_flag[i]) 
        {
            final_boxes.push_back(boxes[i]);
        }
    }
    return final_boxes;
}
