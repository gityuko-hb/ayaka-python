#include <cstdio>
#include <cuda_runtime.h>
#include <torch/extension.h>

static int get_device_count() {
  int count = 0;
  cudaGetDeviceCount(&count);
  return count;
}

static std::string get_device_name(int device) {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  return std::string(prop.name);
}

static torch::Tensor add_one(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  return input + 1.0f;
}

PYBIND11_MODULE(_C, m) {
  m.def("get_device_count", &get_device_count, "Number of CUDA devices");
  m.def("get_device_name", &get_device_name, "CUDA device name");
  m.def("add_one", &add_one, "Add 1.0f to each element");
}
