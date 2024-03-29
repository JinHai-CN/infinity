//
// Created by jinhai on 23-9-27.
//

#include "knn_flat_l2_top1_blas.h"
#include "common/utility/infinity_assert.h"
#include "storage/knnindex/common/distance.h"

#define FINTEGER int

int sgemm_(const char *transa,
           const char *transb,
           FINTEGER *m,
           FINTEGER *n,
           FINTEGER *k,
           const float *alpha,
           const float *a,
           FINTEGER *lda,
           const float *b,
           FINTEGER *ldb,
           float *beta,
           float *c,
           FINTEGER *ldc);

namespace infinity {

template <typename DistType>
void KnnFlatL2Top1Blas<DistType>::Begin() {
    if (begin_ || this->query_count_ == 0) {
        return;
    }

    // block sizes
    const size_t bs_x = faiss::distance_compute_blas_query_bs;
    const size_t bs_y = faiss::distance_compute_blas_database_bs;
    // const size_t bs_x = 16, bs_y = 16;

    ip_block_ = MakeUnique<DistType[]>(bs_x * bs_y);
    x_norms_ = MakeUnique<DistType[]>(this->query_count_);

    fvec_norms_L2sqr(x_norms_.get(), queries_, this->dimension_, this->query_count_);

    for (size_t i0 = 0; i0 < this->query_count_; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > this->query_count_)
            i1 = this->query_count_;

        single_best_result_handler_->begin_multiple(i0, i1);
    }
    begin_ = true;
}

template <typename DistType>
void KnnFlatL2Top1Blas<DistType>::Search(const DistType *base, i16 base_count, i32 segment_id, i16 block_id) {
    if (!begin_) {
        ExecutorError("KnnFlatL2Top1Blas isn't begin")
    }

    this->total_base_count_ += base_count;

    if (base_count == 0) {
        return;
    }

    y_norms_ = MakeUnique<DistType[]>(base_count);
    fvec_norms_L2sqr(y_norms_.get(), base, this->dimension_, base_count);

    // block sizes
    const size_t bs_x = faiss::distance_compute_blas_query_bs;
    const size_t bs_y = faiss::distance_compute_blas_database_bs;

    for (size_t i0 = 0; i0 < this->query_count_; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > this->query_count_)
            i1 = this->query_count_;

        for (i16 j0 = 0; j0 < base_count; j0 += bs_y) {
            i16 j1 = j0 + bs_y;
            if (j1 > base_count)
                j1 = base_count;
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = this->dimension_;
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       base + j0 * this->dimension_,
                       &di,
                       queries_ + i0 * this->dimension_,
                       &di,
                       &zero,
                       ip_block_.get(),
                       &nyi);
            }
            for (int64_t i = i0; i < i1; i++) {
                DistType *ip_line = ip_block_.get() + (i - i0) * (j1 - j0);

                for (size_t j = j0; j < j1; j++) {
                    DistType ip = *ip_line;
                    DistType dis = x_norms_[i] + y_norms_[j] - 2 * ip;

                    // negative values can occur for identical vectors
                    // due to roundoff errors
                    if (dis < 0)
                        dis = 0;

                    *ip_line = dis;
                    ip_line++;
                }
            }
            single_best_result_handler_->add_results(i0, i1, j0, j1, ip_block_.get(), segment_id, block_id);
        }
    }
}

template <typename DistType>
void KnnFlatL2Top1Blas<DistType>::End() {
    if (!begin_)
        return;

    for (i32 i = 0; i < this->query_count_; ++i) {
        single_best_result_handler_->end_multiple();
    }

    begin_ = false;
}

} // namespace infinity
