#包含该目录下的所有源文件
aux_source_directory(. SRC_FILES)
# aux_source_directory(./trt_utils SRC_FILES)


#可执行文件
add_executable(detection ${SRC_FILES})

# 包含头文件路径
target_include_directories(detection PRIVATE
    ${TENSORRT_INCLUDE_DIR}             # TensorRT 头文件路径
    ${CUDA_INCLUDE_DIR}                 # CUDA 头文件路径
    ${OpenCV_INCLUDE_DIRS}              # OpenCV 头文件路径
    ${PROJECT_SOURCE_DIR}/include       # 项目其他头文件路径
)

#连接库（）
target_link_libraries(detection PRIVATE ${TENSORRT_LIB} ${OpenCV_LIBS} ${CUDART_LIB} -ldl -pthread)
