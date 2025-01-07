#ifndef _TENSOR_SCOPE_H_
#define _TENSOR_SCOPE_H_

#include <cstdint>

typedef void (*tensor_callback_t)(uint64_t ptr, int64_t alloc_size, int64_t total_allocated,
                                         int64_t total_reserved);

extern "C" void register_tensor_scope(tensor_callback_t tensor_malloc_callback_ptr,
                                      tensor_callback_t tensor_free_callback_ptr);


extern "C" void tensor_scope_enable();


#endif // _TENSOR_SCOPE_H_
