cmake_minimum_required(VERSION 3.10)
project(weed_detection)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 可执行文件输出目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
# 查找OpenCV库和头文件路径
find_package(OpenCV REQUIRED)

# 查找 TensorRT、CUDA相关的头文件
# 不写路径默认搜索系统路径和环境变量指定的路径
# cuda_runtime.h和cuda_runtime_api.h在同一个目录下，所以只查找其中一个即可
find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h)
find_path(CUDA_INCLUDE_DIR NAMES cuda_runtime_api.h
	PATHS
		/usr/local/cuda/include
		/usr/local/cuda/targets/aarch64-linux/include
)

# 查找库文件,默认搜索系统路径和环境变量指定的路径
find_library(TENSORRT_LIB NAMES nvinfer)
find_library(CUDART_LIB NAMES cudart
	PATHS
		/usr/local/cuda/targets/aarch64-linux/lib
)


# 包含 src 目录下的 CMakeLists.txt
add_subdirectory(src)
