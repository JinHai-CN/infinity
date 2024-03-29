//
// Created by jinhai on 23-6-4.
//

module;

import base_entry;
import stl;
import table_collection_type;
import table_collection_entry;
import parser;
import third_party;
import table_collection_detail;
import table_collection_meta;
import buffer_manager;
import txn_manager;

export module db_entry;

namespace infinity {

//class TxnManager;

export class DBEntry : public BaseEntry {
public:
    inline explicit DBEntry(const SharedPtr<String> &data_dir, SharedPtr<String> db_name, u64 txn_id, TxnTimeStamp begin_ts)
        : BaseEntry(EntryType::kDatabase), db_entry_dir_(MakeShared<String>(Format("{}/{}/txn_{}", *data_dir, *db_name, txn_id))),
          db_name_(Move(db_name)) {
        begin_ts_ = begin_ts;
        txn_id_ = txn_id;
    }

public:
    static EntryResult CreateTableCollection(DBEntry *db_entry,
                                             TableCollectionType table_collection_type,
                                             const SharedPtr<String> &table_collection_name,
                                             const Vector<SharedPtr<ColumnDef>> &columns,
                                             u64 txn_id,
                                             TxnTimeStamp begin_ts,
                                             TxnManager *txn_mgr);

    static EntryResult DropTableCollection(DBEntry *db_entry,
                                           const String &table_collection_name,
                                           ConflictType conflict_type,
                                           u64 txn_id,
                                           TxnTimeStamp begin_ts,
                                           TxnManager *txn_mgr);

    static EntryResult GetTableCollection(DBEntry *db_entry, const String &table_collection_name, u64 txn_id, TxnTimeStamp begin_ts);

    static void RemoveTableCollectionEntry(DBEntry *db_entry, const String &table_collection_name, u64 txn_id, TxnManager *txn_mgr);

    static Vector<TableCollectionEntry *> TableCollections(DBEntry *db_entry, u64 txn_id, TxnTimeStamp begin_ts);

    static Vector<TableCollectionDetail> GetTableCollectionsDetail(DBEntry *db_entry, u64 txn_id, TxnTimeStamp);

    static SharedPtr<String> ToString(DBEntry *db_entry);

    static Json Serialize(const DBEntry *db_entry);

    static UniquePtr<DBEntry> Deserialize(const Json &db_entry_json, BufferManager *buffer_mgr);

public:
    RWMutex rw_locker_{};
    SharedPtr<String> db_entry_dir_{};
    SharedPtr<String> db_name_{};
    HashMap<String, UniquePtr<TableCollectionMeta>> tables_{};
};

} // namespace infinity
