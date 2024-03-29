//
// Created by jinhai on 22-12-24.
//

#include "base_test.h"
#include "main/infinity.h"
#include "function/cast/geography_cast.h"

class PolygonCastTest : public BaseTest {
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

TEST_F(PolygonCastTest, polygon_cast0) {
    using namespace infinity;
    LOG_TRACE("Test name: {}.{}", test_info_->test_case_name(), test_info_->name());

    // Try to cast circle type to wrong type.
    {
        PointT p1(static_cast<f64>(1) + 0.1f, static_cast<f64>(1) - 0.3f);
        PointT p2(static_cast<f64>(1) + 0.5f, static_cast<f64>(1) - 0.7f);
        PointT p3(static_cast<f64>(1) + 0.2f, static_cast<f64>(1) - 0.4f);
        PointT p4(static_cast<f64>(1) + 0.6f, static_cast<f64>(1) - 0.8f);
        PolygonT source;
        source.Initialize(4);
        source.SetPoint(0, p1);
        source.SetPoint(1, p2);
        source.SetPoint(2, p3);
        source.SetPoint(3, p4);
        TinyIntT target;
        EXPECT_THROW(GeographyTryCastToVarlen::Run(source, target, nullptr), FunctionException);
    }
    {
        PointT p1(static_cast<f64>(1) + 0.1f, static_cast<f64>(1) - 0.3f);
        PointT p2(static_cast<f64>(1) + 0.5f, static_cast<f64>(1) - 0.7f);
        PointT p3(static_cast<f64>(1) + 0.2f, static_cast<f64>(1) - 0.4f);
        PointT p4(static_cast<f64>(1) + 0.6f, static_cast<f64>(1) - 0.8f);
        PolygonT source;
        source.Initialize(4);
        source.SetPoint(0, p1);
        source.SetPoint(1, p2);
        source.SetPoint(2, p3);
        source.SetPoint(3, p4);

        VarcharT target;

        SharedPtr<DataType> data_type = MakeShared<DataType>(LogicalType::kVarchar);
        SharedPtr<ColumnVector> col_varchar_ptr = MakeShared<ColumnVector>(data_type);
        col_varchar_ptr->Initialize();

        EXPECT_THROW(GeographyTryCastToVarlen::Run(source, target, col_varchar_ptr), NotImplementException);
    }
}

TEST_F(PolygonCastTest, polygon_cast1) {
    using namespace infinity;
    LOG_TRACE("Test name: {}.{}", test_info_->test_case_name(), test_info_->name());

    // Call BindGeographyCast with wrong type of parameters
    {
        DataType source_type(LogicalType::kPolygon);
        DataType target_type(LogicalType::kDecimal);
        EXPECT_THROW(BindGeographyCast<PolygonT>(source_type, target_type), TypeException);
    }

    SharedPtr<DataType> source_type = MakeShared<DataType>(LogicalType::kPolygon);
    SharedPtr<ColumnVector> col_source = MakeShared<ColumnVector>(source_type);
    col_source->Initialize();
    for (i64 i = 0; i < DEFAULT_VECTOR_SIZE; ++i) {
        PointT p1(static_cast<f64>(i) + 0.1f, static_cast<f64>(i) - 0.3f);
        PointT p2(static_cast<f64>(i) + 0.5f, static_cast<f64>(i) - 0.7f);
        PointT p3(static_cast<f64>(i) + 0.2f, static_cast<f64>(i) - 0.4f);
        PointT p4(static_cast<f64>(i) + 0.6f, static_cast<f64>(i) - 0.8f);
        PolygonT polygon;
        polygon.Initialize(4);
        polygon.SetPoint(0, p1);
        polygon.SetPoint(1, p2);
        polygon.SetPoint(2, p3);
        polygon.SetPoint(3, p4);
        Value v = Value::MakePolygon(polygon);
        col_source->AppendValue(v);
        Value vx = col_source->GetValue(i);
    }
    for (i64 i = 0; i < DEFAULT_VECTOR_SIZE; ++i) {
        PointT p1(static_cast<f64>(i) + 0.1f, static_cast<f64>(i) - 0.3f);
        PointT p2(static_cast<f64>(i) + 0.5f, static_cast<f64>(i) - 0.7f);
        PointT p3(static_cast<f64>(i) + 0.2f, static_cast<f64>(i) - 0.4f);
        PointT p4(static_cast<f64>(i) + 0.6f, static_cast<f64>(i) - 0.8f);
        BoxT bounding_box(PointT(static_cast<f64>(i) + 0.1f, static_cast<f64>(i) - 0.3f),
                          PointT(static_cast<f64>(i) + 0.6f, static_cast<f64>(i) - 0.8f));

        Value vx = col_source->GetValue(i);
        EXPECT_EQ(vx.type().type(), LogicalType::kPolygon);
        EXPECT_EQ(vx.value_.polygon.point_count, 4);
        EXPECT_EQ(*((PointT *)(vx.value_.polygon.ptr)), p1);
        EXPECT_EQ(*((PointT *)(vx.value_.polygon.ptr) + 1), p2);
        EXPECT_EQ(*((PointT *)(vx.value_.polygon.ptr) + 2), p3);
        EXPECT_EQ(*((PointT *)(vx.value_.polygon.ptr) + 3), p4);
        EXPECT_DOUBLE_EQ(vx.value_.polygon.bounding_box.upper_left.x, bounding_box.upper_left.x);
        EXPECT_DOUBLE_EQ(vx.value_.polygon.bounding_box.upper_left.y, bounding_box.upper_left.y);
        EXPECT_DOUBLE_EQ(vx.value_.polygon.bounding_box.lower_right.x, bounding_box.lower_right.x);
        EXPECT_DOUBLE_EQ(vx.value_.polygon.bounding_box.lower_right.y, bounding_box.lower_right.y);
    }
    // cast circle column vector to varchar column vector
    {
        SharedPtr<DataType> target_type = MakeShared<DataType>(LogicalType::kVarchar);
        auto source2target_ptr = BindGeographyCast<PolygonT>(*source_type, *target_type);
        EXPECT_NE(source2target_ptr.function, nullptr);

        SharedPtr<ColumnVector> col_target = MakeShared<ColumnVector>(target_type);
        col_target->Initialize();

        CastParameters cast_parameters;
        EXPECT_THROW(source2target_ptr.function(col_source, col_target, DEFAULT_VECTOR_SIZE, cast_parameters), NotImplementException);
    }
}
