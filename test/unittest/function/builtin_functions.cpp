//
// Created by jinhai on 22-12-24.
//
#include "base_test.h"
#include "main/infinity.h"

class BuiltinFunctionsTest : public BaseTest {
    void SetUp() override {
        infinity::GlobalResourceUsage::Init();
        std::shared_ptr<std::string> config_path = nullptr;
        infinity::Infinity::instance().Init(config_path);
    }

    void TearDown() override {
        infinity::Infinity::instance().UnInit();
        EXPECT_EQ(infinity::GlobalResourceUsage::GetObjectCount(), 0);
        EXPECT_EQ(infinity::GlobalResourceUsage::GetRawMemoryCount(), 0);
        infinity::GlobalResourceUsage::UnInit();
    }
};

TEST_F(BuiltinFunctionsTest, test1) {
    //    using namespace infinity;
    LOG_TRACE("Test name: {}.{}", test_info_->test_case_name(), test_info_->name());
    //
    //    UniquePtr<NewCatalog> catalog_ptr = MakeUnique<NewCatalog>(nullptr);
    //
    //    BuiltinFunctions builtin_functions(catalog_ptr);
    //    builtin_functions.Init();
}