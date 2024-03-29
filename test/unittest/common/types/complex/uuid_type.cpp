//
// Created by JinHai on 2022/11/16.
//

#include "base_test.h"
#include "main/infinity.h"

class UuidTypeTest : public BaseTest {
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

TEST_F(UuidTypeTest, Uuid) {
    using namespace infinity;
    LOG_TRACE("Test name: {}.{}", test_info_->test_case_name(), test_info_->name());
    char uuid_str[17] = "aabbccddeeffgghh";

    // Default constructor and Set
    UuidType uuid1{};
    uuid1.Set(uuid_str);
    EXPECT_STREQ(uuid1.ToString().c_str(), uuid_str);

    // Copy constructor
    UuidType uuid2(uuid1);
    EXPECT_STREQ(uuid2.ToString().c_str(), uuid_str);

    // Copy assignment
    UuidType uuid3{};
    uuid3 = uuid2;
    EXPECT_STREQ(uuid3.ToString().c_str(), uuid_str);

    // Move assignment
    UuidType uuid4{};
    uuid4 = std::move(uuid2);
    EXPECT_STREQ(uuid4.ToString().c_str(), uuid_str);
    EXPECT_STREQ(uuid2.ToString().c_str(), "");

    // Move constructor
    UuidType uuid5(std::move(uuid3));
    EXPECT_STREQ(uuid5.ToString().c_str(), uuid_str);
    EXPECT_STREQ(uuid3.ToString().c_str(), "");

    EXPECT_EQ(uuid1, uuid5);
    EXPECT_FALSE(uuid1 != uuid5);

    // Reset
    uuid5.Reset();
    EXPECT_STREQ(uuid5.ToString().c_str(), "");
}