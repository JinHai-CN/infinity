//
// Created by jinhai on 23-9-27.
//

#include "knn_flat_l2_top1.h"
#include "common/utility/infinity_assert.h"

namespace infinity {

template <typename DistType>
void KnnFlatL2Top1<DistType>::Begin() {
    if (begin_ || this->query_count_ == 0) {
        return;
    }

    for (SizeT i = 0; i < this->query_count_; ++i) {
        single_result_handler_->begin(i);
    }

    begin_ = true;
}

template <typename DistType>
void KnnFlatL2Top1<DistType>::Search(const DistType *base, i16 base_count, i32 segment_id, i16 block_id) {
    if (!begin_) {
        ExecutorError("KnnFlatL2Top1 isn't begin")
    }

    this->total_base_count_ += base_count;

    if (base_count == 0) {
        return;
    }

    for (int64_t i = 0; i < this->query_count_; i++) {
        const DistType *x_i = queries_ + i * this->dimension_;
        const DistType *y_j = base;

        for (i16 j = 0; j < base_count; j++, y_j += this->dimension_) {

            DistType l2_distance = faiss::fvec_L2sqr(x_i, y_j, this->dimension_);
            single_result_handler_->add_result(l2_distance, RowID{segment_id, block_id, j}, i);
        }
    }
}

template <typename DistType>
void KnnFlatL2Top1<DistType>::End() {
    if (!begin_)
        return;

    for (i32 i = 0; i < this->query_count_; ++i) {
        single_result_handler_->end(i);
    }

    begin_ = false;
}

} // namespace infinity
