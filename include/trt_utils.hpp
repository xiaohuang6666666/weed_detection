#pragma once

#include <NvInfer.h>
#include <string>
#include <vector>

class TrtLogger : public nvinfer1::ILogger {
public:
	void log(Severity severity, const char* msg) noexcept override;
};

std::vector<char> readEngine(const std::string& enginePath);
std::string dimsToString(const nvinfer1::Dims& dims);
int inspectEngine(const std::string& enginePath);
