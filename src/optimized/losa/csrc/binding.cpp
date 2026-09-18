#include <torch/extension.h>
void run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor p,
         torch::Tensor o,torch::Tensor l,int64_t n,double scale,int64_t share);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);}
