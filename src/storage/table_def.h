//
// Created by jinhai on 22-12-25.
//

#pragma once

#include "parser/statement/extra/create_table_info.h"
#include "storage/index_def/index_def.h"

namespace infinity {

class TableDef {
public:
    static inline SharedPtr<TableDef> Make(SharedPtr<String> schema, SharedPtr<String> table_name, Vector<SharedPtr<ColumnDef>> columns) {
        return MakeShared<TableDef>(std::move(schema), std::move(table_name), std::move(columns));
    }

public:
    explicit TableDef(SharedPtr<String> schema, SharedPtr<String> table_name, Vector<SharedPtr<ColumnDef>> columns)
        : schema_name_(std::move(schema)), table_name_(std::move(table_name)), columns_(std::move(columns)) {
        SizeT column_count = columns_.size();
        for (SizeT idx = 0; idx < column_count; ++idx) {
            column_name2id_[columns_[idx]->name()] = idx;
        }
    }

    bool operator==(const TableDef &other) const;

    bool operator!=(const TableDef &other) const;

    // Estimated serialized size in bytes, ensured be no less than Write requires, allowed be larger.
    int32_t GetSizeInBytes() const;

    // Write to a char buffer
    void WriteAdv(char *&ptr) const;
    // Read from a serialized version
    static SharedPtr<TableDef> ReadAdv(char *&ptr, int32_t maxbytes);

    [[nodiscard]] inline const Vector<SharedPtr<ColumnDef>> &columns() const { return columns_; }

    [[nodiscard]] inline SizeT column_count() const { return columns_.size(); }

    [[nodiscard]] inline const SharedPtr<String> &table_name() const { return table_name_; }

    [[nodiscard]] inline const SharedPtr<String> &schema_name() const { return schema_name_; }

    [[nodiscard]] inline SizeT GetColIdByName(const String &name) const {
        if (column_name2id_.contains(name)) {
            return column_name2id_.at(name);
        } else {
            return -1;
        }
    }

    String ToString() const;

    void UnionWith(const SharedPtr<TableDef> &other);

private:
    Vector<SharedPtr<ColumnDef>> columns_{};
    HashMap<String, SizeT> column_name2id_{};
    SharedPtr<String> table_name_{};
    SharedPtr<String> schema_name_{};

    Vector<IndexDef> indexes_{};
};

} // namespace infinity
