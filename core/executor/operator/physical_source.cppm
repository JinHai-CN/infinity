//
// Created by jinhai on 23-4-25.
//

module;

import stl;
import parser;
import query_context;
import operator_state;
import physical_operator;
import physical_operator_type;

export module physical_source;

namespace infinity {

export enum class SourceType {
    kInvalid,
    kTable,
    kLocalQueue,
    kEmpty,
    kRemote,
};

export class PhysicalSource final : public PhysicalOperator {
public:
    explicit PhysicalSource(u64 id, SourceType source_type, SharedPtr<Vector<String>> names, SharedPtr<Vector<SharedPtr<DataType>>> types)
        : PhysicalOperator(PhysicalOperatorType::kSource, nullptr, nullptr, id), type_(source_type), output_names_(Move(names)),
          output_types_(Move(types)) {}

    ~PhysicalSource() override = default;

    void Init() override;

    void Execute(QueryContext *query_context) final;

    virtual void Execute(QueryContext *query_context, InputState *input_state, OutputState *output_state) final;

    void Execute(QueryContext *query_context, SourceState *source_state);

    bool ReadyToExec(SourceState *source_state);

    inline SharedPtr<Vector<String>> GetOutputNames() const final { return output_names_; }

    inline SharedPtr<Vector<SharedPtr<DataType>>> GetOutputTypes() const final { return output_types_; }

    inline SourceType source_type() const { return type_; }

private:
    SharedPtr<Vector<String>> output_names_{};
    SharedPtr<Vector<SharedPtr<DataType>>> output_types_{};
    SourceType type_{SourceType::kInvalid};
};

} // namespace infinity
