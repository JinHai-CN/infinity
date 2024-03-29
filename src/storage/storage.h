//
// Created by JinHai on 2022/9/14.
//

#pragma once

#include "main/config.h"
#include "meta/catalog.h"
#include "txn/txn_manager.h"
#include "storage/buffer/buffer_manager.h"
#include "storage/wal/wal_manager.h"

namespace infinity {

class Storage {
public:
    explicit Storage(const Config *config_ptr);

    [[nodiscard]] inline NewCatalog *catalog() noexcept { return new_catalog_.get(); }

    [[nodiscard]] inline BufferManager *buffer_manager() noexcept { return buffer_mgr_.get(); }

    [[nodiscard]] inline TxnManager *txn_manager() const noexcept { return txn_mgr_.get(); }

    void Init();

    void UnInit();

private:
    static SharedPtr<DirEntry> GetLatestCatalog(const String &dir);

    static void InitCatalog(NewCatalog *catalog, TxnManager *txn_mgr);

private:
    const Config *config_ptr_{};
    UniquePtr<NewCatalog> new_catalog_{};
    UniquePtr<BufferManager> buffer_mgr_{};
    UniquePtr<TxnManager> txn_mgr_{};
    UniquePtr<WalManager> wal_mgr_{};
};

} // namespace infinity
