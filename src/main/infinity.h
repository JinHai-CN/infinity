//
// Created by JinHai on 2022/7/19.
//

#pragma once

#include "common/singleton.h"

#include "resource_manager.h"
#include "scheduler/fragment_scheduler.h"
#include "storage/storage.h"

namespace infinity {

class Config;

class Infinity : public Singleton<Infinity> {
public:
    [[nodiscard]] inline FragmentScheduler *fragment_scheduler() noexcept { return fragment_scheduler_.get(); }

    [[nodiscard]] inline Config *config() noexcept { return config_.get(); }

    [[nodiscard]] inline Storage *storage() noexcept { return storage_.get(); }

    [[nodiscard]] inline ResourceManager *resource_manager() noexcept { return resource_manager_.get(); }

    void Init(const SharedPtr<String> &config_path);

    void UnInit();

private:
    friend class Singleton;

    Infinity() = default;

    UniquePtr<Config> config_{};
    UniquePtr<ResourceManager> resource_manager_{};
    UniquePtr<FragmentScheduler> fragment_scheduler_{};
    UniquePtr<Storage> storage_{};

    bool initialized_{false};
};

} // namespace infinity
