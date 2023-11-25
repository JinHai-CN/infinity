// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module;

#include <ctime>

import stl;
import index_def;
import base_entry;
import table_index_meta;
import table_collection_entry;
import third_party;
import local_file_system;
import default_values;
import random;
import base_index;
import parser;
import infinity_exception;
import column_index_entry;
import segment_column_index_entry;
import irs_index_entry;

module table_index_entry;

namespace infinity {

TableIndexEntry::TableIndexEntry(SharedPtr<IndexDef> index_def,
                                 TableIndexMeta *table_index_meta,
                                 SharedPtr<String> index_dir,
                                 u64 txn_id,
                                 TxnTimeStamp begin_ts)
    : BaseEntry(EntryType::kTableIndex), table_index_meta_(table_index_meta), index_def_(Move(index_def)), index_dir_(Move(index_dir)) {
    begin_ts_ = begin_ts; // TODO:: begin_ts and txn_id should be const and set in BaseEntry
    txn_id_ = txn_id;

    SizeT index_count = index_def->index_array_.size();
    column_index_map_.reserve(index_count);
    for (SizeT idx = 0; idx < index_count; ++idx) {
        SharedPtr<BaseIndex> &base_index = index_def->index_array_[idx];

        // Get column info
        Assert<StorageException>(base_index->column_names_.size() == 1, "Currently, composite index doesn't supported.");
        u64 column_id = TableIndexMeta::GetTableCollectionEntry(table_index_meta)->GetColumnIdByName(base_index->column_names_[0]);
        SharedPtr<String> column_index_path = MakeShared<String>(Format("{}/{}", *index_dir_, base_index->column_names_[0]));
        SharedPtr<ColumnIndexEntry> column_index_entry =
            ColumnIndexEntry::NewColumnIndexEntry(base_index, column_id, this, txn_id, column_index_path, begin_ts);
        column_index_map_[idx] = column_index_entry;
    }
}

UniquePtr<TableIndexEntry>
TableIndexEntry::NewTableIndexEntry(SharedPtr<IndexDef> index_def, TableIndexMeta *table_index_meta, u64 txn_id, TxnTimeStamp begin_ts) {
    SharedPtr<String> index_dir =
        DetermineIndexDir(*TableIndexMeta::GetTableCollectionEntry(table_index_meta)->table_entry_dir_, *index_def->index_name_);
    return MakeUnique<TableIndexEntry>(Move(index_def), table_index_meta, Move(index_dir), txn_id, begin_ts);
}

void TableIndexEntry::CommitCreateIndex(TableIndexEntry *table_index_entry,
                                        u64 column_id,
                                        u32 segment_id,
                                        SharedPtr<SegmentColumnIndexEntry> segment_column_index_entry) {
    UniqueLock<RWMutex> w_locker(table_index_entry->rw_locker_);
    ColumnIndexEntry* column_index_entry = table_index_entry->column_index_map_[column_id].get();
    column_index_entry->index_by_segment.emplace(segment_id, segment_column_index_entry);
}

void TableIndexEntry::CommitCreateIndex(TableIndexEntry *table_index_entry, SharedPtr<IrsIndexEntry> irs_index_entry) {
    table_index_entry->irs_index_entry_ = irs_index_entry;
}

Json TableIndexEntry::Serialize(const TableIndexEntry *table_index_entry, TxnTimeStamp max_commit_ts) {
    Json json;
    json["txn_id"] = table_index_entry->txn_id_.load();
    json["begin_ts"] = table_index_entry->begin_ts_;
    json["commit_ts"] = table_index_entry->commit_ts_.load();
    json["deleted"] = table_index_entry->deleted_;
    if (!table_index_entry->deleted_) {
        json["index_dir"] = *table_index_entry->index_dir_;
        //        json["base_index"] = table_index_entry->base_index_->Serialize();
        //        for (const auto &[segment_id, index_entry] : table_index_entry->index_by_segment) {
        //            ColumnIndexEntry::Flush(index_entry.get(), max_commit_ts);
        //            json["indexes"].push_back(SegmentColumnIndexEntry::Serialize(index_entry.get()));
        //        }
    }
    return json;
}

UniquePtr<TableIndexEntry> TableIndexEntry::Deserialize(const Json &index_def_entry_json,
                                                        TableIndexMeta *table_index_meta,
                                                        BufferManager *buffer_mgr,
                                                        TableCollectionEntry *table_entry) {
    u64 txn_id = index_def_entry_json["txn_id"];
    TxnTimeStamp begin_ts = index_def_entry_json["begin_ts"];
    TxnTimeStamp commit_ts = index_def_entry_json["commit_ts"];
    bool deleted = index_def_entry_json["deleted"];

    if (deleted) {
        auto table_index_entry = MakeUnique<TableIndexEntry>(nullptr, table_index_meta, nullptr, txn_id, begin_ts);
        table_index_entry->deleted_ = true;
        table_index_entry->commit_ts_.store(commit_ts);
        return table_index_entry;
    }

    auto index_dir = MakeShared<String>(index_def_entry_json["index_dir"]);
    //    SharedPtr<BaseIndex> base_index = BaseIndex::Deserialize(index_def_entry_json["base_index"]);
    //
    //    auto table_index_entry = MakeUnique<TableIndexEntry>(base_index, table_index_meta, index_dir, txn_id, begin_ts);
    //    table_index_entry->commit_ts_.store(commit_ts);
    //    table_index_entry->deleted_ = deleted;
    //
    //    if (index_def_entry_json.contains("indexes")) {
    //        for (const auto &index_entry_json : index_def_entry_json["indexes"]) {
    //            u32 segment_id = index_entry_json["segment_id"];
    //            SegmentEntry *segment_entry = table_entry->segment_map_.at(segment_id).get();
    //
    //            if (base_index->column_names_.size() > 1) {
    //                StorageException("Not implemented.");
    //            }
    //            u64 column_id = table_entry->GetColumnIdByName(base_index->column_names_[0]);
    //            SharedPtr<ColumnDef> column_def = table_entry->column_map_[column_id];
    //
    //            UniquePtr<CreateIndexParam> create_index_param = SegmentEntry::GetCreateIndexParam(segment_entry, base_index, column_def);
    //
    //            auto index_entry =
    //                SegmentColumnIndexEntry::Deserialize(index_entry_json, table_index_entry.get(), segment_entry, buffer_mgr,
    //                Move(create_index_param));
    //            table_index_entry->index_by_segment.emplace(index_entry->segment_entry_->segment_id_, index_entry);
    //        }
    //    }

    //    return table_index_entry;
}

SharedPtr<String> TableIndexEntry::DetermineIndexDir(const String &parent_dir, const String &index_name) {
    LocalFileSystem fs;
    SharedPtr<String> index_dir;
    do {
        u32 seed = time(nullptr);
        index_dir = MakeShared<String>(Format("{}/{}_index_{}", parent_dir, RandomString(DEFAULT_RANDOM_NAME_LEN, seed), index_name));
    } while (!fs.CreateDirectoryNoExp(*index_dir));
    return index_dir;
}

} // namespace infinity
