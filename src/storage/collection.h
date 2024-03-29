//
// Created by jinhai on 23-3-16.
//

#pragma once

#include "base_table.h"

namespace infinity {

class Collection : public BaseTable {
public:
    explicit Collection(SharedPtr<String> schema_name, SharedPtr<String> collection_name)
        : BaseTable(TableCollectionType::kCollectionEntry, std::move(schema_name), std::move(collection_name)) {}

    [[nodiscard]] inline SizeT row_count() const { return row_count_; }

private:
    SizeT row_count_{0};
};

} // namespace infinity
