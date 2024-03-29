//
// Created by jinhai on 23-6-23.
//

module;

#include <vector>

import base_entry;
import table_collection_type;
import stl;
import parser;
import txn_manager;
import table_collection_entry;
import table_collection_detail;
import table_collection_meta;
import buffer_manager;
import default_values;
import block_index;
import logger;
import third_party;

module db_entry;

namespace infinity {

EntryResult DBEntry::CreateTableCollection(DBEntry *db_entry,
                                           TableCollectionType table_collection_type,
                                           const SharedPtr<String> &table_collection_name,
                                           const Vector<SharedPtr<ColumnDef>> &columns,
                                           u64 txn_id,
                                           TxnTimeStamp begin_ts,
                                           TxnManager *txn_mgr) {
    const String &table_name = *table_collection_name;

    // Check if there is table_meta with the table_name
    db_entry->rw_locker_.lock_shared();

    TableCollectionMeta *table_meta{nullptr};
    if (db_entry->tables_.find(table_name) != db_entry->tables_.end()) {
        table_meta = db_entry->tables_[table_name].get();
    }
    db_entry->rw_locker_.unlock_shared();

    // no table_meta
    if (table_meta == nullptr) {
        // Create new db meta
        LOG_TRACE(Format("Create new table/collection: {}", table_name));
        UniquePtr<TableCollectionMeta> new_table_meta = MakeUnique<TableCollectionMeta>(db_entry->db_entry_dir_, table_collection_name, db_entry);
        table_meta = new_table_meta.get();

        db_entry->rw_locker_.lock();
        db_entry->tables_[table_name] = Move(new_table_meta);
        db_entry->rw_locker_.unlock();

        LOG_TRACE(Format("Add new table entry for {} in new table meta of db_entry {} ", table_name, *db_entry->db_entry_dir_));

        EntryResult res =
            TableCollectionMeta::CreateNewEntry(table_meta, table_collection_type, table_collection_name, columns, txn_id, begin_ts, txn_mgr);
        return res;
    } else {
        LOG_TRACE(Format("Add new table entry for {} in existed table meta of db_entry {}", table_name, *db_entry->db_entry_dir_));
        EntryResult res =
            TableCollectionMeta::CreateNewEntry(table_meta, table_collection_type, table_collection_name, columns, txn_id, begin_ts, txn_mgr);
        return res;
    }
}

EntryResult DBEntry::DropTableCollection(DBEntry *db_entry,
                                         const String &table_collection_name,
                                         ConflictType conflict_type,
                                         u64 txn_id,
                                         TxnTimeStamp begin_ts,
                                         TxnManager *txn_mgr) {
    db_entry->rw_locker_.lock_shared();

    TableCollectionMeta *table_meta{nullptr};
    if (db_entry->tables_.find(table_collection_name) != db_entry->tables_.end()) {
        table_meta = db_entry->tables_[table_collection_name].get();
    }
    db_entry->rw_locker_.unlock_shared();
    if (table_meta == nullptr) {
        if (conflict_type == ConflictType::kIgnore) {
            LOG_TRACE(Format("Ignore drop a not existed table/collection entry {}", table_collection_name));
            return {nullptr, nullptr};
        }
        LOG_TRACE(Format("Attempt to drop not existed table/collection entry {}", table_collection_name));
        return {nullptr, MakeUnique<String>("Attempt to drop not existed table/collection entry")};
    }

    LOG_TRACE(Format("Drop a table/collection entry {}", table_collection_name));
    EntryResult res = TableCollectionMeta::DropNewEntry(table_meta, txn_id, begin_ts, txn_mgr, table_collection_name, conflict_type);

    return res;
}

EntryResult DBEntry::GetTableCollection(DBEntry *db_entry, const String &table_name, u64 txn_id, TxnTimeStamp begin_ts) {
    db_entry->rw_locker_.lock_shared();

    TableCollectionMeta *table_meta{nullptr};
    if (db_entry->tables_.find(table_name) != db_entry->tables_.end()) {
        table_meta = db_entry->tables_[table_name].get();
    }
    db_entry->rw_locker_.unlock_shared();

    //    LOG_TRACE("Get a table entry {}", table_name);
    if (table_meta == nullptr) {
        return {nullptr, MakeUnique<String>("Attempt to get not existed table/collection entry")};
    }
    return TableCollectionMeta::GetEntry(table_meta, txn_id, begin_ts);
}

void DBEntry::RemoveTableCollectionEntry(DBEntry *db_entry, const String &table_collection_name, u64 txn_id, TxnManager *txn_mgr) {
    db_entry->rw_locker_.lock_shared();

    TableCollectionMeta *table_meta{nullptr};
    if (db_entry->tables_.find(table_collection_name) != db_entry->tables_.end()) {
        table_meta = db_entry->tables_[table_collection_name].get();
    }
    db_entry->rw_locker_.unlock_shared();

    LOG_TRACE(Format("Remove a table/collection entry: {}", table_collection_name));
    TableCollectionMeta::DeleteNewEntry(table_meta, txn_id, txn_mgr);
}

Vector<TableCollectionEntry *> DBEntry::TableCollections(DBEntry *db_entry, u64 txn_id, TxnTimeStamp begin_ts) {
    Vector<TableCollectionEntry *> results;

    db_entry->rw_locker_.lock_shared();

    results.reserve(db_entry->tables_.size());
    for (auto &table_collection_meta_pair : db_entry->tables_) {
        TableCollectionMeta *table_collection_meta = table_collection_meta_pair.second.get();
        EntryResult entry_result = TableCollectionMeta::GetEntry(table_collection_meta, txn_id, begin_ts);
        if (entry_result.entry_ == nullptr) {
            LOG_WARN(Format("error when get table/collection entry: {}", *entry_result.err_));
        } else {
            if (entry_result.entry_ != nullptr) {
                results.emplace_back((TableCollectionEntry *)entry_result.entry_);
            }
        }
    }
    db_entry->rw_locker_.unlock_shared();

    return results;
}

Vector<TableCollectionDetail> DBEntry::GetTableCollectionsDetail(DBEntry *db_entry, u64 txn_id, TxnTimeStamp begin_ts) {

    Vector<TableCollectionEntry *> table_collection_entries = DBEntry::TableCollections(db_entry, txn_id, begin_ts);
    Vector<TableCollectionDetail> results;
    results.reserve(table_collection_entries.size());
    for (TableCollectionEntry *table_collection_entry : table_collection_entries) {
        TableCollectionDetail table_collection_detail;
        table_collection_detail.db_name_ = db_entry->db_name_;
        table_collection_detail.table_collection_name_ = table_collection_entry->table_collection_name_;
        table_collection_detail.table_collection_type_ = table_collection_entry->table_collection_type_;
        table_collection_detail.column_count_ = table_collection_entry->columns_.size();
        table_collection_detail.row_count_ = table_collection_entry->row_count_;
        table_collection_detail.segment_capacity_ = DEFAULT_SEGMENT_CAPACITY;

        SharedPtr<BlockIndex> segment_index = TableCollectionEntry::GetBlockIndex(table_collection_entry, txn_id, begin_ts);

        table_collection_detail.segment_count_ = segment_index->SegmentCount();
        table_collection_detail.block_count_ = segment_index->BlockCount();
        results.emplace_back(table_collection_detail);
    }

    return results;
}

SharedPtr<String> DBEntry::ToString(DBEntry *db_entry) {
    SharedLock<RWMutex> r_locker(db_entry->rw_locker_);
    SharedPtr<String> res = MakeShared<String>(
        Format("DBEntry, db_entry_dir: {}, txn id: {}, table count: ", *db_entry->db_entry_dir_, db_entry->txn_id_, db_entry->tables_.size()));
    return res;
}

Json DBEntry::Serialize(const DBEntry *db_entry) {
    Json json_res;

    json_res["db_entry_dir"] = *db_entry->db_entry_dir_;
    json_res["db_name"] = *db_entry->db_name_;
    json_res["txn_id"] = db_entry->txn_id_.load();
    json_res["begin_ts"] = db_entry->begin_ts_;
    json_res["commit_ts"] = db_entry->commit_ts_.load();
    json_res["deleted"] = db_entry->deleted_;
    json_res["entry_type"] = db_entry->entry_type_;
    for (const auto &table_meta_pair : db_entry->tables_) {
        json_res["tables"].emplace_back(TableCollectionMeta::Serialize(table_meta_pair.second.get()));
    }
    return json_res;
}

UniquePtr<DBEntry> DBEntry::Deserialize(const Json &db_entry_json, BufferManager *buffer_mgr) {
    Json json_res;

    SharedPtr<String> db_entry_dir = MakeShared<String>(db_entry_json["db_entry_dir"]);
    SharedPtr<String> db_name = MakeShared<String>(db_entry_json["db_name"]);
    u64 txn_id = db_entry_json["txn_id"];
    u64 begin_ts = db_entry_json["begin_ts"];
    UniquePtr<DBEntry> res = MakeUnique<DBEntry>(db_entry_dir, db_name, txn_id, begin_ts);

    u64 commit_ts = db_entry_json["commit_ts"];
    bool deleted = db_entry_json["deleted"];
    EntryType entry_type = db_entry_json["entry_type"];
    res->commit_ts_ = commit_ts;
    res->deleted_ = deleted;
    res->entry_type_ = entry_type;

    if (db_entry_json.contains("tables")) {
        for (const auto &table_meta_json : db_entry_json["tables"]) {
            UniquePtr<TableCollectionMeta> table_meta = TableCollectionMeta::Deserialize(table_meta_json, res.get(), buffer_mgr);
            res->tables_.emplace(*table_meta->table_collection_name_, Move(table_meta));
        }
    }

    return res;
}

} // namespace infinity
