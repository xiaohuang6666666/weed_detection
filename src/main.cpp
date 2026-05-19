#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>
#include <cctype>
// #include "trt_utils.hpp"    // 包含检查引擎文件的工具
#include "yolov5.hpp"

void detect_image(const std::string& enginePath, const std::string& imagePath) 
{
    YOLOv5 model(enginePath);
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) 
    {
        printf("read image error!\n");
        return;
    }
    // 推理可检测框坐标
    auto boxes = model.detect(img);
    printf("图片检测完成！目标数量：%d\n", (int)boxes.size());
    // 画框
    for (auto& box : boxes) 
    {
        cv::rectangle(img,
            cv::Point(box.x1, box.y1),
            cv::Point(box.x2, box.y2),
            cv::Scalar(0, 255, 0), 2);
    }
    // 保存结果
    cv::imwrite("/home/jetson/weed_detection/results/result_image.jpg", img);
    printf("图片检测结果已保存为 results/result_image.jpg\n");
}

void detect_video(const std::string& enginePath, const std::string& videoPath) 
{
    YOLOv5 model(enginePath);
    cv::VideoCapture cap = cv::VideoCapture(videoPath);
    if (!cap.isOpened()) 
    {
        printf("read video error!\n");
        return;
    }
    // 原视频信息
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));         // 视频宽度
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));       // 视频高度
    double fps = cap.get(cv::CAP_PROP_FPS);                                  // 视频帧率
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));  // 视频总帧数

    // 避免输入视频帧率读取失败
    if (fps <= 1e-3) {
        std::cout << "[WARN] 输入视频 FPS 读取失败，回退到 30 FPS" << std::endl;
        fps = 30.0;
    }

    // 设置输出视频信息（与输入视频一致）
    std::string output_path = "/home/jetson/weed_detection/results/result_video.mp4";
    cv::VideoWriter writer(
        output_path,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cout << "open writer error!" << std::endl;
        return;
    }

    std::cout << "[INFO] 输入信息: " << width << "x" << height
              << ", fps=" << fps
              << ", total_frames=" << total_frames << std::endl;

    // 视频推理 + 画框
    cv::Mat frame;
    int frame_cnt = 0;
    int written_frames = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (cap.read(frame))
    {
        frame_cnt++;
        // 推理
        auto boxes = model.detect(frame);
        for (auto& box : boxes) 
        {
            // 检测框
            cv::rectangle(frame,
                cv::Point(box.x1, box.y1),
                cv::Point(box.x2, box.y2),
                cv::Scalar(0, 255, 0), 2);
            // 标签
            std::string label = cv::format("weed: %.2f", box.score);
            cv::putText(frame, label, 
                cv::Point(box.x1, box.y1 - 5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                cv::Scalar(0, 255, 0), 1);
        }

        // 每帧只写入一次，避免重复写帧导致输出视频时长被拉长
        writer.write(frame);
        written_frames++;

        // 打印处理进度和推理速度
        if (frame_cnt % 30 == 0 || frame_cnt == total_frames) 
        {
            auto now = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - t0).count();
            double proc_fps = (elapsed_s > 1e-6) ? (frame_cnt / elapsed_s) : 0.0;
            std::cout << "\r✅ 处理进度: " << (float)frame_cnt / total_frames * 100
                      << "% | 推理速度: " << proc_fps << " FPS" << std::flush;
        }
    }

    // 打印输出信息
    std::cout << std::endl;
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    double avg_proc_fps = (elapsed_s > 1e-6) ? (frame_cnt / elapsed_s) : 0.0;
    double output_duration_s = (fps > 1e-6) ? (written_frames / fps) : 0.0;
    std::cout << "[INFO] 输出信息: written_frames=" << written_frames
              << ", output_duration=" << output_duration_s << "s"
              << ", process_time=" << elapsed_s << "s"
              << ", avg_proc_fps=" << avg_proc_fps << std::endl;

    cap.release();
    writer.release();
    printf("视频检测结果已保存为 results/result_video.mp4\n"); 
}

void detect_camera(const std::string& enginePath, int device = 0)
{
    YOLOv5 model(enginePath);
    cv::VideoCapture cap(device);
    if (!cap.isOpened())
    {
        printf("open camera error!\n");
        return;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double cam_fps = cap.get(cv::CAP_PROP_FPS);
    if (cam_fps <= 1e-3) cam_fps = 30.0;

    std::cout << "[INFO] 摄像头打开: " << width << "x" << height << ", fps=" << cam_fps << std::endl;

    cv::Mat frame;
    int frame_cnt = 0;
    auto t0 = std::chrono::steady_clock::now();
    double proc_fps = 0.0;

    const std::string window_name = "Camera Detection";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    while (true)
    {
        if (!cap.read(frame))
        {
            std::cout << "read frame error!" << std::endl;
            break;
        }
        frame_cnt++;

        // 推理
        auto boxes = model.detect(frame);
        for (auto& box : boxes)
        {
            cv::rectangle(frame,
                cv::Point(box.x1, box.y1),
                cv::Point(box.x2, box.y2),
                cv::Scalar(0, 255, 0), 2);
            std::string label = cv::format("weed: %.2f", box.score);
            cv::putText(frame, label,
                cv::Point(box.x1, box.y1 - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1);
        }

        // 计算并显示处理帧率
        auto now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(now - t0).count();
        if (elapsed_s > 1e-6) proc_fps = frame_cnt / elapsed_s;
        cv::putText(frame, cv::format("proc FPS: %.2f", proc_fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::imshow(window_name, frame);

        // 按 ESC 退出
        int key = cv::waitKey(1);
        if (key == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
	if (argc < 2) 
    {
		std::cout << "Usage: " << argv[0] << " <engine_path>" << std::endl;
		std::cout << "Example: " << argv[0] << " mode/best.engine" << std::endl;
		return 1;
	}
    const std::string enginePath = argv[1];
    const std::string imagePath = "/home/jetson/weed_detection/test/test_image.jpg";
    const std::string videoPath = "/home/jetson/weed_detection/test/test_video.mp4";
    // 图片检测
    detect_image(enginePath, imagePath);
    // 相机检测
    detect_camera(enginePath);
    // 视频检测
    // detect_video(enginePath, videoPath);
    // return inspectEngine(enginePath);    // 检查引擎文件
    return 0;
}
