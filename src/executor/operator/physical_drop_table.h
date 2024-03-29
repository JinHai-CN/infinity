//
// Created by JinHai on 2022/7/28.
//

#pragma once

#include "executor/physical_operator.h"
#include "parser/statement/extra/extra_ddl_info.h"

namespace infinity {

class PhysicalDropTable final : public PhysicalOperator {
public:
    explicit PhysicalDropTable(SharedPtr<String> schema_name,
                               SharedPtr<String> tbl_name,
                               ConflictType conflict_type,
                               SharedPtr<Vector<String>> output_names,
                               SharedPtr<Vector<SharedPtr<DataType>>> output_types,
                               u64 id)
        : PhysicalOperator(PhysicalOperatorType::kDropTable, nullptr, nullptr, id), schema_name_(std::move(schema_name)),
          table_name_(std::move(tbl_name)), conflict_type_(conflict_type), output_names_(std::move(output_names)),
          output_types_(std::move(output_types)) {}

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
