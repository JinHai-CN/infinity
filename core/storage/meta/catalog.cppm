//
// Created by jinhai on 23-6-4.
//

module;

import stl;
import base_entry;
import db_entry;
import db_meta;
import txn_manager;
import function;
import function_set;
import table_function;
import third_party;
import buffer_manager;

export module new_catalog;

namespace infinity {

//class Txn;
//class TxnManager;
//class FunctionSet;
//class TableFunction;
//class DBEntry;
//class BufferManager;

export struct NewCatalog {
public:
    explicit NewCatalog(SharedPtr<String> dir);

public:
    static EntryResult CreateDatabase(NewCatalog *catalog, const String &db_name, u64 txn_id, TxnTimeStamp begin_ts, TxnManager *txn_mgr);

    static EntryResult DropDatabase(NewCatalog *catalog, const String &db_name, u64 txn_id, TxnTimeStamp begin_ts, TxnManager *txn_mgr);

    static EntryResult GetDatabase(NewCatalog *catalog, const String &db_name, u64 txn_id, TxnTimeStamp begin_ts);

    static void RemoveDBEntry(NewCatalog *catalog, const String &db_name, u64 txn_id, TxnManager *txn_mgr);

    static Vector<DBEntry *> Databases(NewCatalog *catalog, u64 txn_id, TxnTimeStamp begin_ts);

    // Function related methods
    static SharedPtr<FunctionSet> GetFunctionSetByName(NewCatalog *catalog, String function_name);

    static void AddFunctionSet(NewCatalog *catalog, const SharedPtr<FunctionSet> &function_set);

    static void DeleteFunctionSet(NewCatalog *catalog, String function_name);

    // Table Function related methods
    static SharedPtr<TableFunction> GetTableFunctionByName(NewCatalog *catalog, String function_name);

    static void AddTableFunction(NewCatalog *catalog, const SharedPtr<TableFunction> &table_function);

    static void DeleteTableFunction(NewCatalog *catalog, String function_name);

    static Json Serialize(const NewCatalog *catalog);

    static void Deserialize(const Json &catalog_json, BufferManager *buffer_mgr, UniquePtr<NewCatalog> &catalog);

    static UniquePtr<NewCatalog> LoadFromFile(const SharedPtr<DirEntry> &dir_entry, BufferManager *buffer_mgr);

    static void SaveAsFile(const NewCatalog *catalog_ptr, const String &dir, const String &file_name);

public:
    SharedPtr<String> current_dir_{nullptr};
    HashMap<String, UniquePtr<DBMeta>> databases_{};
    u64 next_txn_id_{};
    u64 catalog_version_{};
    RWMutex rw_locker_;

    // Currently, these function or function set can't be changed and also will not be persistent.
    HashMap<String, SharedPtr<FunctionSet>> function_sets_;
    HashMap<String, SharedPtr<TableFunction>> table_functions_;
};

} // namespace infinity
