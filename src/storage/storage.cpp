//
// Created by jinhai on 23-10-16.
//

module;

#include <filesystem>
#include <functional>
#include <regex>
#include <string>

import base_entry;
import config;
import stl;
import buffer_manager;
import default_values;
import wal_manager;
import new_catalog;
import txn_manager;
import builtin_functions;
import local_file_system;
import third_party;
import logger;
import parser;
import txn;
import infinity_exception;
import infinity_assert;

module storage;

namespace infinity {

Storage::Storage(const Config *config_ptr) : config_ptr_(config_ptr) {}

void Storage::Init() {
    // Construct buffer manager
    buffer_mgr_ = MakeUnique<BufferManager>(config_ptr_->buffer_pool_size(), config_ptr_->data_dir(), config_ptr_->temp_dir());

    // Check the data dir to get latest catalog file.
    String catalog_dir = String(*config_ptr_->data_dir()) + "/" + String(CATALOG_FILE_DIR);

    // Construct wal manager
    wal_mgr_ = MakeUnique<WalManager>(this,
                                      Path(*config_ptr_->wal_dir()) / WAL_FILE_TEMP_FILE,
                                      config_ptr_->wal_size_threshold(),
                                      config_ptr_->full_checkpoint_interval_sec(),
                                      config_ptr_->delta_checkpoint_interval_sec(),
                                      config_ptr_->delta_checkpoint_interval_wal_bytes());

    // Must init catalog before txn manager.
    // Replay wal file wrap init catalog
    auto start_time_stamp = wal_mgr_->ReplayWalFile();
    start_time_stamp = (!exist_catalog_) ? 1 : start_time_stamp + 1;
    // Construct txn manager
    txn_mgr_ = MakeUnique<TxnManager>(new_catalog_.get(),
                                      buffer_mgr_.get(),
                                      std::bind(&WalManager::PutEntry, wal_mgr_.get(), std::placeholders::_1),
                                      0,
                                      start_time_stamp);
    txn_mgr_->Start();
    // start WalManager after TxnManager since it depends on TxnManager.
    wal_mgr_->Start();

    BuiltinFunctions builtin_functions(new_catalog_);
    builtin_functions.Init();
}

void Storage::UnInit() {
    Printf("Shutdown storage ...\n");
    txn_mgr_->Stop();
    wal_mgr_->Stop();

    txn_mgr_.reset();
    wal_mgr_.reset();

    new_catalog_.reset();
    buffer_mgr_.reset();
    config_ptr_ = nullptr;
    Printf("Shutdown storage successfully\n");
}

SharedPtr<DirEntry> Storage::GetLatestCatalog(const String &dir) {
    LocalFileSystem fs;
    if (!fs.Exists(dir)) {
        fs.CreateDirectory(dir);
        return nullptr;
    }

    Vector<SharedPtr<DirEntry>> dir_array = fs.ListDirectory(dir);
    SharedPtr<DirEntry> latest;
    int64_t latest_version_number = -1;
    const std::regex catalog_file_regex("META_[0-9]+\\.full.json");
    for (const auto &dir_entry_ptr : dir_array) {
        LOG_TRACE(Format("Candidate file name: {}", dir_entry_ptr->path().c_str()));
        if (dir_entry_ptr->is_regular_file()) {
            String current_file_name = dir_entry_ptr->path().filename();
            if (std::regex_match(current_file_name, catalog_file_regex)) {
                int64_t version_number = std::stoll(current_file_name.substr(5, current_file_name.size() - 10));
                if (version_number > latest_version_number) {
                    latest_version_number = version_number;
                    latest = dir_entry_ptr;
                }
            }
        }
    }

    return latest;
}

void Storage::InitCatalog(NewCatalog *catalog, TxnManager *txn_mgr) {
    EntryResult create_res;
    Txn *new_txn = txn_mgr->CreateTxn();
    new_txn->BeginTxn();
    create_res = new_txn->CreateDatabase("default", ConflictType::kError);
    new_txn->CommitTxn();
    if (create_res.err_ != nullptr) {
        Error<StorageException>(*create_res.err_, __FILE_NAME__, __LINE__);
    }
}

void Storage::AttachCatalog(const Vector<String> &catalog_files) {
    LOG_INFO(Format("Attach catalogs from {} files", catalog_files.size()));
    for (const auto &catalog_file : catalog_files) {
        LOG_TRACE(Format("Catalog file: {}", catalog_file.c_str()));
    }
    new_catalog_ = NewCatalog::LoadFromFiles(catalog_files, buffer_mgr_.get());
    exist_catalog_ = true;
}

void Storage::InitNewCatalog() {
    LOG_INFO("Init new catalog");
    String catalog_dir = String(*config_ptr_->data_dir()) + "/" + String(CATALOG_FILE_DIR);
    LocalFileSystem fs;
    if (!fs.Exists(catalog_dir)) {
        fs.CreateDirectory(catalog_dir);
    }
    new_catalog_ = MakeUnique<NewCatalog>(MakeShared<String>(catalog_dir), true);
    exist_catalog_ = false;
}

} // namespace infinity
