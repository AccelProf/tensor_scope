#include "tensor_scope.h"
#include "torch_prof.h"


void register_tensor_scope(tensor_callback_t tensor_malloc_callback_ptr,
                           tensor_callback_t tensor_free_callback_ptr) {
    TorchProf& torch_prof = TorchProf::getInstance();
    torch_prof.register_tensor_callback(tensor_malloc_callback_ptr, tensor_free_callback_ptr);
}

void tensor_scope_enable() {
    TorchProf& torch_prof = TorchProf::getInstance();
    torch_prof.enable_torch_callback();
}
