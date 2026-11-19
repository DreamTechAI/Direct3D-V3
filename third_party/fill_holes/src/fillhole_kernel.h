#pragma once
#include <torch/extension.h>

torch::Tensor triangulate_loops_cuda(const std::vector<std::vector<int32_t>>& loops, const torch::Tensor& vertices, const std::vector<int64_t>& edges_keys);
