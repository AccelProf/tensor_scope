#include "torch_operator.h"

TorchOperator& TorchOperator::getInstance() {
    static TorchOperator instance;
    return instance;
}

void TorchOperator::enable_torch_callback() {
    auto callback = at::RecordFunctionCallback(
            &TorchOperator::operator_start_callback,
            &TorchOperator::operator_end_callback
        ).scopes({at::RecordScope::FUNCTION});
    operator_callback_handle_ = at::addGlobalCallback(callback);
}

void TorchOperator::register_operator_callback(TorchScopeType_t scope_type,
                                         operator_callback_t callback_ptr) {
    if (scope_type == TORCH_SCOPE_OPERATOR_START) {
        this->operator_start_callback_ptr = callback_ptr;
    } else if (scope_type == TORCH_SCOPE_OPERATOR_END) {
        this->operator_end_callback_ptr = callback_ptr;
    }
}

std::unique_ptr<at::ObserverContext> TorchOperator::operator_start_callback(const at::RecordFunction& fn) {
    if (getInstance().operator_start_callback_ptr == nullptr) {
        printf("operator_start_callback_ptr is nullptr\n");
    }
    getInstance().operator_start_callback_ptr(fn.name());
    return nullptr;
}

void TorchOperator::operator_end_callback(const at::RecordFunction& fn, at::ObserverContext* ctx) {
    if (getInstance().operator_end_callback_ptr == nullptr) {
        printf("operator_end_callback_ptr is nullptr\n");
    }
    getInstance().operator_end_callback_ptr(fn.name());
}
