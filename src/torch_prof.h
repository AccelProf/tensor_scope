#ifndef _TORCH_PROF_H_
#define _TORCH_PROF_H_

#include <torch/extension.h>
#include "tensor_scope.h"


class TorchProf {
public:
    void enable_torch_callback();

    void register_tensor_callback(tensor_callback_t tensor_malloc_callback_ptr,
                             tensor_callback_t tensor_free_callback_ptr);

    void tensor_malloc_callback(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                int64_t total_reserved);

    void tensor_free_callback(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                int64_t total_reserved);

    static TorchProf& getInstance();

private:
    TorchProf() {}
    ~TorchProf() {}

    class TorchCallback : public c10::MemoryReportingInfoBase {
    public:
        TorchCallback() {}

        bool memoryProfilingEnabled() const override { return true; }

#if TORCH_VERSION_MAJOR >= 2
        void reportMemoryUsage(void* ptr, int64_t alloc_size, size_t total_allocated,
                                size_t total_reserved, c10::Device device) override;
#else
        void reportMemoryUsage(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                int64_t total_reserved, c10::Device device) override;
#endif
    };  // class TorchCallback

    std::shared_ptr<TorchCallback> new_torch_profiler() {
        return std::make_shared<TorchCallback>();
    }

    tensor_callback_t tensor_malloc_callback_ptr = nullptr;
    tensor_callback_t tensor_free_callback_ptr = nullptr;
};  // class TorchProf


#endif //_TORCH_PROF_H_
