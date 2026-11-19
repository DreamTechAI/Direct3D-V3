#include <torch/extension.h>

void compute_valid_sdf_cuda(float* vertices, int* faces, bool* voxel, const int numTriangles, const int DIM=512, const float threshold=8);

extern "C"
void compute_valid_sdf_wrapper(torch::Tensor vertices, torch::Tensor faces, torch::Tensor voxel, const int numTriangles, const int DIM=512, const float threshold=8.0) {
    compute_valid_sdf_cuda(vertices.data_ptr<float>(), faces.data_ptr<int>(), voxel.data_ptr<bool>(), numTriangles, DIM, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize", &compute_valid_sdf_wrapper, "Compute SDF using CUDA");
}
