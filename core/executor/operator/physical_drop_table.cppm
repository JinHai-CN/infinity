//
// Created by JinHai on 2022/7/28.
//

module;

import stl;
import parser;
import query_context;
import operator_state;
import physical_operator;
import physical_operator_type;

export module physical_drop_table;

namespace infinity {

export class PhysicalDropTable final : public PhysicalOperator {
public:
    explicit PhysicalDropTable(SharedPtr<String> schema_name,
                               SharedPtr<String> tbl_name,
                               ConflictType conflict_type,
                               SharedPtr<Vector<String>> output_names,
                               SharedPtr<Vector<SharedPtr<DataType>>> output_types,
                               u64 id)
        : PhysicalOperator(PhysicalOperatorType::kDropTable, nullptr, nullptr, id), schema_name_(Move(schema_name)),
          table_name_(Move(tbl_name)), conflict_type_(conflict_type), output_names_(Move(output_names)),
          output_types_(Move(output_types)) {}

    ~PhysicalDropTable() override = default;

    void Init() override;

    void Execute(QueryContext *query_context) final;

    virtual void Execute(QueryContext *query_context, InputState *input_state, OutputState *output_state) final;

    inline SharedPtr<Vector<String>> GetOutputNames() const final { return output_names_; }

    inline SharedPtr<Vector<SharedPtr<DataType>>> GetOutputTypes() const final { return output_types_; }

    inline SharedPtr<String> schema_name() const { return schema_name_; }

    inline SharedPtr<String> table_name() const { return table_name_; }

    inline ConflictType conflict_type() const { return conflict_type_; }

private:
    SharedPtr<String> schema_name_{};
    SharedPtr<String> table_name_{};
    ConflictType conflict_type_{ConflictType::kInvalid};

    SharedPtr<Vector<String>> output_names_{};
    SharedPtr<Vector<SharedPtr<DataType>>> output_types_{};
};

} // namespace infinity
