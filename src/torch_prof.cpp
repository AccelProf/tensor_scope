#include "torch_prof.h"

TorchProf& TorchProf::getInstance() {
    static TorchProf instance;
    return instance;
}

void TorchProf::enable_torch_callback() {
    auto profiler = new_torch_profiler();
    c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, profiler);
}

void TorchProf::register_tensor_callback(tensor_callback_t tensor_malloc_callback_ptr,
                                         tensor_callback_t tensor_free_callback_ptr) {
    this->tensor_malloc_callback_ptr = tensor_malloc_callback_ptr;
    this->tensor_free_callback_ptr = tensor_free_callback_ptr;
}

void TorchProf::tensor_malloc_callback(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                        int64_t total_reserved) {
    // yosemite_tensor_malloc_callback(ptr, alloc_size, total_allocated, total_reserved);
    if (this->tensor_malloc_callback_ptr == nullptr) {
        printf("tensor_malloc_callback_ptr is nullptr\n");
        return;
    }
    this->tensor_malloc_callback_ptr(ptr, alloc_size, total_allocated, total_reserved);
}

void TorchProf::tensor_free_callback(void* ptr, int64_t alloc_size,
                                        int64_t total_allocated, int64_t total_reserved) {
    // yosemite_tensor_free_callback(ptr, alloc_size, total_allocated, total_reserved);
    if (this->tensor_free_callback_ptr == nullptr) {
        printf("tensor_free_callback_ptr is nullptr\n");
        return;
    }
    this->tensor_free_callback_ptr(ptr, alloc_size, total_allocated, total_reserved);
}

#if TORCH_VERSION_MAJOR >= 2
void TorchProf::TorchCallback::reportMemoryUsage(void* ptr, int64_t alloc_size, size_t total_allocated,
                                                    size_t total_reserved, c10::Device device) {
    if (alloc_size > 0) {
        TorchProf::getInstance().tensor_malloc_callback(ptr, alloc_size, total_allocated, total_reserved);
    } else {
        TorchProf::getInstance().tensor_free_callback(ptr, alloc_size, total_allocated, total_reserved);
    }
}
#else
void TorchProf::TorchCallback::reportMemoryUsage(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                                    int64_t total_reserved, c10::Device device) {
    if (alloc_size > 0) {
        TorchProf::getInstance().tensor_malloc_callback(ptr, alloc_size, total_allocated, total_reserved);
    } else {
        TorchProf::getInstance().tensor_free_callback(ptr, alloc_size, total_allocated, total_reserved);
    }
}
#endif
