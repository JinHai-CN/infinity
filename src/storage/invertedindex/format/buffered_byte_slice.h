#pragma once

#include "common/memory/byte_slice.h"
#include "common/memory/memory_pool.h"
#include "flush_info.h"
#include "posting_value.h"
#include "short_buffer.h"
#include "storage/invertedindex/index_defines.h"
#include "storage/io/byte_slice_writer.h"

namespace infinity {

class BufferedByteSlice {
public:
    BufferedByteSlice(MemoryPool *byte_slice_pool, MemoryPool *buffer_pool);

    virtual ~BufferedByteSlice() = default;

    void Init(const PostingValues *value);

    template <typename T>
    void PushBack(uint8_t row, T value);

    void EndPushBack() {
        flush_info_.SetIsValidShortBuffer(true);
        buffer_.EndPushBack();
    }

    bool NeedFlush(uint8_t need_flush_count = MAX_DOC_PER_RECORD) const { return buffer_.Size() == need_flush_count; }

    const ByteSliceList *GetByteSliceList() const { return posting_writer_.GetByteSliceList(); }

    MemoryPool *GetBufferPool() const { return buffer_.GetPool(); }

    const PostingValues *GetPostingValues() const { return buffer_.GetPostingValues(); }

    void SnapShot(BufferedByteSlice *buffer) const;

    bool IsShortBufferValid() const { return flush_info_.IsValidShortBuffer(); }

    const ShortBuffer &GetBuffer() const { return buffer_; }

    size_t GetBufferSize() const { return buffer_.Size(); }

    size_t GetTotalCount() const { return flush_info_.GetFlushCount() + buffer_.Size(); }

    FlushInfo GetFlushInfo() const { return flush_info_; }

    size_t Flush();

    virtual void Dump(const std::shared_ptr<FileWriter> &file) { posting_writer_.Dump(file); }

    virtual size_t EstimateDumpSize() const { return posting_writer_.GetSize(); }

protected:
    virtual size_t DoFlush();

protected:
    FlushInfo flush_info_;
    ShortBuffer buffer_;
    ByteSliceWriter posting_writer_;

    friend class BufferedByteSliceTest;
};

template <typename T>
inline void BufferedByteSlice::PushBack(uint8_t row, T value) {
    buffer_.PushBack(row, value);
}

} // namespace infinity