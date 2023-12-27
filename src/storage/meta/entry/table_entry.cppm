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

import stl;
import parser;
import txn_store;
import buffer_manager;
import third_party;
import table_entry_type;
import base_entry;
import segment_entry;
import block_index;
import data_access_state;
import txn_manager;
import segment_column_index_entry;
import segment_entry;
import table_index_meta;
import txn;
import status;

export module table_entry;

namespace infinity {

class DBEntry;
class IndexDef;
class TableIndexEntry;
class IrsIndexEntry;
class TableMeta;

export struct TableEntry : public BaseEntry {
public:
    // for iterator unit test.
    explicit TableEntry() : BaseEntry(EntryType::kTable) {}

    explicit TableEntry(const SharedPtr<String> &db_entry_dir,
                        SharedPtr<String> table_collection_name,
                        const Vector<SharedPtr<ColumnDef>> &columns,
                        TableEntryType table_entry_type,
                        TableMeta *table_meta,
                        u64 txn_id,
                        TxnTimeStamp begin_ts);

public:
    Tuple<TableIndexEntry *, Status>
    CreateIndex(const SharedPtr<IndexDef> &index_def, ConflictType conflict_type, u64 txn_id, TxnTimeStamp begin_ts, TxnManager *txn_mgr);

    Tuple<TableIndexEntry *, Status>
    DropIndex(const String &index_name, ConflictType conflict_type, u64 txn_id, TxnTimeStamp begin_ts, TxnManager *txn_mgr);

    static Status
    GetIndex(TableEntry *table_entry, const String &index_name, u64 txn_id, TxnTimeStamp begin_ts, BaseEntry *&segment_column_index_entry);

    static void RemoveIndexEntry(TableEntry *table_entry, const String &index_name, u64 txn_id, TxnManager *txn_mgr);

public:
    static void Append(TableEntry *table_entry, Txn *txn_ptr, void *txn_store, BufferManager *buffer_mgr);

    static void
    CreateIndexFile(TableEntry *table_entry, void *txn_store, TableIndexEntry *table_index_entry, TxnTimeStamp begin_ts, BufferManager *buffer_mgr);

    static UniquePtr<String> Delete(TableEntry *table_entry, Txn *txn_ptr, DeleteState &delete_state);

    static void CommitAppend(TableEntry *table_entry, Txn *txn_ptr, const AppendState *append_state_ptr);

    static void CommitCreateIndex(TableEntry *table_entry, HashMap<String, TxnIndexStore> &txn_indexes_store_);

    static void RollbackAppend(TableEntry *table_entry, Txn *txn_ptr, void *txn_store);

    static void CommitDelete(TableEntry *table_entry, Txn *txn_ptr, const DeleteState &append_state);

    static UniquePtr<String> RollbackDelete(TableEntry *table_entry, Txn *txn_ptr, DeleteState &append_state, BufferManager *buffer_mgr);

    static UniquePtr<String> ImportSegment(TableEntry *table_entry, Txn *txn_ptr, SharedPtr<SegmentEntry> segment);

    static inline u32 GetNextSegmentID(TableEntry *table_entry) { return table_entry->next_segment_id_++; }

    static inline u32 GetMaxSegmentID(const TableEntry *table_entry) { return table_entry->next_segment_id_; }

    static SegmentEntry *GetSegmentByID(const TableEntry *table_entry, u32 seg_id);

    static DBEntry *GetDBEntry(const TableEntry *table_entry);

    static SharedPtr<String> GetDBName(const TableEntry *table_entry);

    inline static TableMeta *GetTableMeta(const TableEntry *table_entry) { return table_entry->table_entry_; }

    static SharedPtr<BlockIndex> GetBlockIndex(TableEntry *table_entry, u64 txn_id, TxnTimeStamp begin_ts);

    static Json Serialize(TableEntry *table_entry, TxnTimeStamp max_commit_ts, bool is_full_checkpoint);

    static UniquePtr<TableEntry> Deserialize(const Json &table_entry_json, TableMeta *table_meta, BufferManager *buffer_mgr);

    static void GetFullTextAnalyzers(TableEntry *table_entry,
                                     u64 txn_id,
                                     TxnTimeStamp begin_ts,
                                     SharedPtr<IrsIndexEntry> &irs_index_entry,
                                     Map<String, String> &column2analyzer);

    virtual void MergeFrom(BaseEntry &other);

public:
    u64 GetColumnIdByName(const String &column_name);

private:
    HashMap<String, u64> column_name2column_id_;

public:
    RWMutex rw_locker_{};

    SharedPtr<String> table_entry_dir_{};

    SharedPtr<String> table_collection_name_{};

    Vector<SharedPtr<ColumnDef>> columns_{};

    TableEntryType table_entry_type_{TableEntryType::kTableEntry};

    TableMeta *table_entry_{};

    // From data table
    Atomic<SizeT> row_count_{};
    Map<u32, SharedPtr<SegmentEntry>> segment_map_{};
    SegmentEntry *unsealed_segment_{};
    atomic_u32 next_segment_id_{};

    // Index definition
    HashMap<String, UniquePtr<TableIndexMeta>> index_meta_map_{};
};

} // namespace infinity
