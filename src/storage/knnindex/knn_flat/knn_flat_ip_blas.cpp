//
// Created by jinhai on 23-9-27.
//

#include "knn_flat_ip_blas.h"

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
void KnnFlatIPBlas<DistType>::Begin() {
    if (begin_ || this->query_count_ == 0) {
        return;
    }

    const SizeT bs_x = faiss::distance_compute_blas_query_bs;
    for (SizeT i0 = 0; i0 < this->query_count_; i0 += bs_x) {
        SizeT i1 = i0 + bs_x;
        if (i1 > this->query_count_)
            i1 = this->query_count_;

        heap_result_handler_->begin_multiple(i0, i1);
    }

    const size_t bs_y = faiss::distance_compute_blas_database_bs;
    ip_block_ = MakeUnique<DistType[]>(bs_x * bs_y);

    begin_ = true;
}

template <typename DistType>
void KnnFlatIPBlas<DistType>::Search(const DistType *base, i16 base_count, i32 segment_id, i16 block_id) {
    if (!begin_) {
        ExecutorError("KnnFlatIPBlas isn't begin")
    }

    this->total_base_count_ += base_count;

    if (base_count == 0) {
        return;
    }

    const SizeT bs_x = faiss::distance_compute_blas_query_bs;
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

            heap_result_handler_->add_results(i0, i1, j0, j1, ip_block_.get(), segment_id, block_id);
        }
    }
}

template <typename DistType>
void KnnFlatIPBlas<DistType>::End() {
    if (!begin_)
        return;

    const SizeT bs_x = faiss::distance_compute_blas_query_bs;
    for (SizeT i0 = 0; i0 < this->query_count_; i0 += bs_x) {
        SizeT i1 = i0 + bs_x;
        if (i1 > this->query_count_)
            i1 = this->query_count_;

        heap_result_handler_->end_multiple(i0, i1);
    }

    begin_ = false;
}

} // namespace infinity
