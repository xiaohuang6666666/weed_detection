#include "../../include/trt_utils.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

std::vector<char> readEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file) {
        return {};
    }

    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

std::string dimsToString(const nvinfer1::Dims& dims) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        oss << dims.d[i];
        if (i != dims.nbDims - 1)
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

int inspectEngine(const std::string& enginePath) {
    auto engineData = readEngine(enginePath);
    if (engineData.empty()) {
        std::cerr << "Failed to read engine\n";
        return 1;
    }

    TrtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(engineData.data(), engineData.size())
    );

    std::cout << "Engine loaded successfully.\n";
    int nTensors = engine->getNbIOTensors();
    std::cout << "I/O tensors: " << nTensors << "\n";

    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());

    for (int i = 0; i < nTensors; ++i) {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        auto engineDims = engine->getTensorShape(name);
        auto runtimeDims = context->getTensorShape(name);

        std::cout << (mode == nvinfer1::TensorIOMode::kINPUT ? "[Input]  " : "[Output] ")
                  << i << " : " << name
                  << " engineShape=" << dimsToString(engineDims)
                  << " runtimeShape=" << dimsToString(runtimeDims)
                  << "\n";
    }

    return 0;
}
