//
// Created by jinhai on 23-9-15.
//

#pragma once

#include "executor/physical_operator.h"

namespace infinity {

class PhysicalMergeParallelAggregate final : public PhysicalOperator {
public:
    explicit PhysicalMergeParallelAggregate(SharedPtr<Vector<String>> output_names, SharedPtr<Vector<SharedPtr<DataType>>> output_types, u64 id)
        : PhysicalOperator(PhysicalOperatorType::kMergeParallelAggregate, nullptr, nullptr, id), output_names_(std::move(output_names)),
          output_types_(std::move(output_types)) {}

    ~PhysicalMergeParallelAggregate() override = default;

    void Init() override;

    void Execute(QueryContext *query_context) final;

    virtual void Execute(QueryContext *query_context, InputState *input_state, OutputState *output_state) final;

    inline SharedPtr<Vector<String>> GetOutputNames() const final { return output_names_; }

    inline SharedPtr<Vector<SharedPtr<DataType>>> GetOutputTypes() const final { return output_types_; }

private:
    SharedPtr<Vector<String>> output_names_{};
    SharedPtr<Vector<SharedPtr<DataType>>> output_types_{};
};

} // namespace infinity
