//
// Created by jinhai on 23-3-16.
//

module;

import stl;
import parser;
import query_context;
import operator_state;
import physical_operator;
import physical_operator_type;

export module physical_drop_collection;

namespace infinity {

export class PhysicalDropCollection : public PhysicalOperator {
public:
    explicit PhysicalDropCollection(SharedPtr<String> schema_name, SharedPtr<String> collection_name, ConflictType conflict_type, u64 id)
        : PhysicalOperator(PhysicalOperatorType::kDropCollection, nullptr, nullptr, id), schema_name_(Move(schema_name)),
          collection_name_(Move(collection_name)), conflict_type_(conflict_type) {}

    ~PhysicalDropCollection() override = default;

    void Init() override;

    void Execute(QueryContext *query_context) final;

    virtual void Execute(QueryContext *query_context, InputState *input_state, OutputState *output_state) final;

    inline SharedPtr<Vector<String>> GetOutputNames() const final { return output_names_; }

    inline SharedPtr<Vector<SharedPtr<DataType>>> GetOutputTypes() const final { return output_types_; }

    inline SharedPtr<String> schema_name() const { return schema_name_; }

    inline SharedPtr<String> collection_name() const { return collection_name_; }

    inline ConflictType conflict_type() const { return conflict_type_; }

private:
    SharedPtr<String> schema_name_{};
    SharedPtr<String> collection_name_{};
    ConflictType conflict_type_{ConflictType::kInvalid};

    SharedPtr<Vector<String>> output_names_{};
    SharedPtr<Vector<SharedPtr<DataType>>> output_types_{};
};

} // namespace infinity
