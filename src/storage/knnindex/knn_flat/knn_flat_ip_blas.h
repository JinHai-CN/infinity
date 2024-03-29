//
// Created by jinhai on 23-9-27.
//

#pragma once

#include "storage/knnindex/knn_distance.h"
#include "storage/knnindex/knn_flat/alias.h"

namespace infinity {

template <typename DistType>
class KnnFlatIPBlas final : public KnnDistance<DistType> {

    using HeapResultHandler = NewHeapResultHandler<faiss::CMin<DistType, RowID>>;

public:
    explicit KnnFlatIPBlas(const DistType *queries, i64 query_count, i64 topk, i64 dimension, EmbeddingDataType elem_data_type)
        : KnnDistance<DistType>(KnnDistanceAlgoType::kKnnFlatIpBlas, elem_data_type, query_count, topk, dimension), queries_(queries) {
        id_array_ = MakeUnique<Vector<RowID>>(topk * this->query_count_, RowID());
        distance_array_ = MakeUnique<DistType[]>(sizeof(DistType) * topk * this->query_count_);

        heap_result_handler_ = MakeUnique<HeapResultHandler>(query_count, distance_array_.get(), id_array_->data(), topk);
    }

    void Begin() final;

    void Search(const DistType *base, i16 base_count, i32 segment_id, i16 block_id) final;

    void End() final;

    [[nodiscard]] inline DistType *GetDistances() const final { return heap_result_handler_->heap_dis_tab; }

    [[nodiscard]] inline RowID *GetIDs() const final { return heap_result_handler_->heap_ids_tab; }

    [[nodiscard]] inline DistType *GetDistanceByIdx(i64 idx) const final {
        if (idx >= this->query_count_) {
            ExecutorError("Query index exceeds the limit")
        }
        return heap_result_handler_->heap_dis_tab + idx * this->top_k_;
    }

    [[nodiscard]] inline RowID *GetIDByIdx(i64 idx) const final {
        if (idx >= this->query_count_) {
            ExecutorError("Query index exceeds the limit")
        }
        return heap_result_handler_->heap_ids_tab + idx * this->top_k_;
    }

private:
    UniquePtr<Vector<RowID>> id_array_{};
    UniquePtr<DistType[]> distance_array_{};
    UniquePtr<DistType[]> ip_block_{};

    UniquePtr<HeapResultHandler> heap_result_handler_{};
    const DistType *queries_{};
    bool begin_{false};
};

template class KnnFlatIPBlas<f32>;

} // namespace infinity
