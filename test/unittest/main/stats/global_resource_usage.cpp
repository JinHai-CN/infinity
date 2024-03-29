//
// Created by JinHai on 2022/11/25.
//

#include "base_test.h"
#include "main/infinity.h"
#include "main/logger.h"

class GlobalResourceUsageTest : public BaseTest {
    void SetUp() override {
        infinity::GlobalResourceUsage::Init();
    }

    void TearDown() override {
        EXPECT_EQ(infinity::GlobalResourceUsage::GetObjectCount(), 0);
        EXPECT_EQ(infinity::GlobalResourceUsage::GetRawMemoryCount(), 0);
        infinity::GlobalResourceUsage::UnInit();
    }
};

TEST_F(GlobalResourceUsageTest, usage_test) {
    using namespace infinity;
    LOG_TRACE("Test name: {}.{}", test_info_->test_case_name(), test_info_->name());

    // Object count
    GlobalResourceUsage::IncrObjectCount();
    EXPECT_EQ(GlobalResourceUsage::GetObjectCount(), 1);

    GlobalResourceUsage::IncrObjectCount();
    EXPECT_EQ(GlobalResourceUsage::GetObjectCount(), 2);

    GlobalResourceUsage::DecrObjectCount();
    EXPECT_EQ(GlobalResourceUsage::GetObjectCount(), 1);

    GlobalResourceUsage::DecrObjectCount();
    EXPECT_EQ(GlobalResourceUsage::GetObjectCount(), 0);

    GlobalResourceUsage::DecrObjectCount();
    EXPECT_EQ(GlobalResourceUsage::GetObjectCount(), -1);

    GlobalResourceUsage::IncrObjectCount();
    EXPECT_EQ(GlobalResourceUsage::GetObjectCount(), 0);

    // Raw memory count
    GlobalResourceUsage::IncrRawMemCount();
    EXPECT_EQ(GlobalResourceUsage::GetRawMemoryCount(), 1);

    GlobalResourceUsage::IncrRawMemCount();
    EXPECT_EQ(GlobalResourceUsage::GetRawMemoryCount(), 2);

    GlobalResourceUsage::DecrRawMemCount();
    EXPECT_EQ(GlobalResourceUsage::GetRawMemoryCount(), 1);

    GlobalResourceUsage::DecrRawMemCount();
    EXPECT_EQ(GlobalResourceUsage::GetRawMemoryCount(), 0);

    GlobalResourceUsage::DecrRawMemCount();
    EXPECT_EQ(GlobalResourceUsage::GetRawMemoryCount(), -1);

    GlobalResourceUsage::IncrRawMemCount();
    EXPECT_EQ(GlobalResourceUsage::GetRawMemoryCount(), 0);
}