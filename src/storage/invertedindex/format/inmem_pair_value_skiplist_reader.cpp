#include "inmem_pair_value_skiplist_reader.h"
#include "main/logger.h"

namespace infinity {
InMemPairValueSkipListReader::InMemPairValueSkipListReader(MemoryPool *session_pool) : session_pool_(session_pool), skiplist_buffer_(nullptr) {}

InMemPairValueSkipListReader::~InMemPairValueSkipListReader() {
    if (session_pool_) {
        skiplist_buffer_->~BufferedByteSlice();
        session_pool_->Deallocate((void *)skiplist_buffer_, sizeof(BufferedByteSlice));
    } else {
        delete skiplist_buffer_;
        skiplist_buffer_ = nullptr;
    }
}

void InMemPairValueSkipListReader::Load(BufferedByteSlice *posting_buffer) {
    skipped_item_count_ = -1;
    current_key_ = 0;
    current_value_ = 0;
    prev_key_ = 0;
    prev_value_ = 0;
    current_cursor_ = 0;
    num_in_buffer_ = 0;

    BufferedByteSlice *skiplist_buffer = new BufferedByteSlice(session_pool_, session_pool_);
    posting_buffer->SnapShot(skiplist_buffer);

    skiplist_buffer_ = skiplist_buffer;
    skiplist_reader_.Open(skiplist_buffer_);
}

std::pair<int, bool> InMemPairValueSkipListReader::LoadBuffer() {
    size_t key_num = 0;
    size_t flush_count = skiplist_buffer_->GetTotalCount();
    size_t decode_count = SKIP_LIST_BUFFER_SIZE;
    if (!skiplist_reader_.Decode(key_buffer_, decode_count, key_num)) {
        return std::make_pair(0, false);
    }
    size_t value_num = 0;
    if (!skiplist_reader_.Decode(value_buffer_, decode_count, value_num)) {
        return std::make_pair(0, false);
    }
    if (key_num != value_num) {
        LOG_ERROR(fmt::format("SKipList decode error, doc_num = {} offset_num = {}", key_num, value_num));
        return std::make_pair(-1, false);
    }
    num_in_buffer_ = key_num;
    current_cursor_ = 0;
    return std::make_pair(0, true);
}
} // namespace infinity