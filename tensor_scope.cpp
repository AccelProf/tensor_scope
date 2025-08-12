#include <torch/extension.h>

#define VERBOSE 1

#if VERBOSE
#define PRINT(...) do { fprintf(stdout, __VA_ARGS__); fflush(stdout); } while (0)
#else
#define PRINT(...)
#endif

class TorchTensor {
public:
    void enable_torch_callback() {
        auto profiler = new_torch_profiler();
        c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, profiler);
    }

    void tensor_malloc_callback(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                int64_t total_reserved, int device_id) {
        PRINT("Malloc tensor %p with size %ld, allocated %ld, reserved %ld on device %d\n",
            (void*)ptr, alloc_size, total_allocated, total_reserved, device_id);
    }

    void tensor_free_callback(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                int64_t total_reserved, int device_id) {
        PRINT("Free tensor %p with size %ld, allocated %ld, reserved %ld on device %d\n",
            (void*)ptr, alloc_size, total_allocated, total_reserved, device_id);
    }

    static TorchTensor& getInstance() {
        static TorchTensor instance;
        return instance;
    }

private:
    TorchTensor() {}
    ~TorchTensor() {}

    class TorchCallback : public c10::MemoryReportingInfoBase {
    public:
        TorchCallback() {}

        bool memoryProfilingEnabled() const override { return true; }

#if TORCH_VERSION_MAJOR >= 2
    void reportMemoryUsage(void* ptr, int64_t alloc_size, size_t total_allocated,
                            size_t total_reserved, c10::Device device) override {
        if (device.is_cuda() || device.is_hip()) {
            if (alloc_size > 0) {
                TorchTensor::getInstance().tensor_malloc_callback(
                    ptr, alloc_size, total_allocated, total_reserved, device.index());
            } else {
                TorchTensor::getInstance().tensor_free_callback(
                    ptr, alloc_size, total_allocated, total_reserved, device.index());
            }
        }
    }
#else
    void reportMemoryUsage(void* ptr, int64_t alloc_size, int64_t total_allocated,
                            int64_t total_reserved, c10::Device device) override {
        if (device.is_cuda() || device.is_hip()) {
            if (alloc_size > 0) {
                TorchTensor::getInstance().tensor_malloc_callback(
                    ptr, alloc_size, total_allocated, total_reserved, device.index());
            } else {
                TorchTensor::getInstance().tensor_free_callback(
                    ptr, alloc_size, total_allocated, total_reserved, device.index());
            }
        }
    }
#endif
    };  // class TorchCallback

    std::shared_ptr<TorchCallback> new_torch_profiler() {
        return std::make_shared<TorchCallback>();
    }
};  // class TorchTensor

int _init_tensor_scope() {
    TorchTensor::getInstance().enable_torch_callback();
    return 0;
}

int _initialization = _init_tensor_scope();
