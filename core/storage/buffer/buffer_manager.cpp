//
// Created by jinhai on 23-7-2.
//

module;

#include "concurrentqueue.h"

import stl;
import buffer_handle;
import logger;
import third_party;
import infinity_assert;
import infinity_exception;
import default_values;
import async_batch_processor;
import buffer_task;
import local_file_system;

module buffer_manager;

namespace infinity {

BufferManager::BufferManager(SizeT mem_limit, SharedPtr<String> base_dir, SharedPtr<String> temp_dir)
    : mem_limit_(mem_limit), base_dir_(Move(base_dir)), temp_dir_(Move(temp_dir)) {}

void BufferManager::Init() {
    reader_ =
        MakeUnique<AsyncBatchProcessor>(DEFAULT_READER_PREPARE_QUEUE_SIZE, DEFAULT_READER_COMMIT_QUEUE_SIZE, BufferIO::OnPrepare, BufferIO::OnCommit);

    writer_ =
        MakeUnique<AsyncBatchProcessor>(DEFAULT_WRITER_PREPARE_QUEUE_SIZE, DEFAULT_WRITER_COMMIT_QUEUE_SIZE, BufferIO::OnPrepare, BufferIO::OnCommit);

    LocalFileSystem fs;
    if (!fs.Exists(*base_dir_)) {
        fs.CreateDirectory(*base_dir_);
    }
    if (!fs.Exists(*temp_dir_)) {
        fs.CreateDirectory(*temp_dir_);
    }
}

void BufferManager::PushGCQueue(BufferHandle *buffer_handle) { queue_.enqueue(buffer_handle); }

BufferHandle *BufferManager::GetBufferHandle(const SharedPtr<String> &file_dir, const SharedPtr<String> &filename, BufferType buffer_type) {
    SharedPtr<String> full_name{};
    if (file_dir.get() == nullptr or file_dir->empty()) {
        full_name = filename;
    } else {
        full_name = MakeShared<String>(Format("{}/{}", *file_dir, *filename));
    }

    {
        SharedLock<RWMutex> r_locker(rw_locker_);
        if (buffer_map_.find(*full_name) != buffer_map_.end()) {
            return &buffer_map_.at(*full_name);
        }
    }

    switch (buffer_type) {
        case BufferType::kTempFile: {
            LOG_ERROR("Temp file meta data should be stored in buffer manager");
            Error<NotImplementException>("Not implemented here", __FILE_NAME__, __LINE__);
            break;
        }
        case BufferType::kFile: {
            LOG_TRACE("Read file, Generate Buffer Handle");
            UniqueLock<RWMutex> w_locker(rw_locker_);
            auto iter = buffer_map_.emplace(*full_name, this);
            iter.first->second.id_ = next_buffer_id_++;
            iter.first->second.base_dir_ = this->base_dir_;
            iter.first->second.temp_dir_ = this->temp_dir_;
            iter.first->second.current_dir_ = file_dir;
            iter.first->second.file_name_ = filename;
            iter.first->second.buffer_type_ = buffer_type;

            return &(iter.first->second);
        }
        case BufferType::kExtraBlock: {
            LOG_TRACE("Read extra block, Generate Buffer Handle");
            UniqueLock<RWMutex> w_locker(rw_locker_);
            auto iter = buffer_map_.emplace(*full_name, this);
            iter.first->second.base_dir_ = this->base_dir_;
            iter.first->second.temp_dir_ = this->temp_dir_;
            iter.first->second.current_dir_ = file_dir;
            iter.first->second.file_name_ = filename;
            iter.first->second.buffer_type_ = buffer_type;

            return &(iter.first->second);
        }
        case BufferType::kInvalid: {
            LOG_ERROR("Buffer type is invalid");
            break;
        }
    }
    return nullptr;
}

BufferHandle *BufferManager::GetBufferHandle(const SharedPtr<String> &file_dir,
                                             const SharedPtr<String> &filename,
                                             offset_t offset,
                                             SizeT buffer_size,
                                             BufferType buffer_type) {
    SharedPtr<String> full_name{};
    if (file_dir.get() == nullptr or file_dir->empty()) {
        full_name = filename;
    } else {
        full_name = MakeShared<String>(Format("{}/{}", *file_dir, *filename));
    }

    {
        SharedLock<RWMutex> r_locker(rw_locker_);
        if (buffer_map_.find(*full_name) != buffer_map_.end()) {
            return &buffer_map_.at(*full_name);
        }
    }

    switch (buffer_type) {
        case BufferType::kTempFile: {
            LOG_ERROR("Temp file meta data should be stored in buffer manager");
            Error<NotImplementException>("Not implemented here", __FILE_NAME__, __LINE__);
        }
        case BufferType::kFile: {
            LOG_TRACE("Read file, Generate Buffer Handle");
            UniqueLock<RWMutex> w_locker(rw_locker_);
            auto iter = buffer_map_.emplace(*full_name, this);
            iter.first->second.id_ = next_buffer_id_++;
            iter.first->second.base_dir_ = this->base_dir_;
            iter.first->second.temp_dir_ = this->temp_dir_;
            iter.first->second.current_dir_ = file_dir;
            iter.first->second.file_name_ = filename;
            iter.first->second.buffer_type_ = buffer_type;
            iter.first->second.offset_ = offset;
            iter.first->second.buffer_size_ = buffer_size;

            return &(iter.first->second);
        }
        case BufferType::kExtraBlock: {
            Error<NotImplementException>("Not implemented here", __FILE_NAME__, __LINE__);
            //            LOG_TRACE("Read extra block, Generate Buffer Handle");
            //            UniqueLock<RWMutex> w_locker(rw_locker_);
            //            auto iter = buffer_map_.emplace(*full_name, this);
            //            iter.first->second.base_dir_ = this->base_dir_;
            //            iter.first->second.temp_dir_ = this->temp_dir_;
            //            iter.first->second.current_dir_ = file_dir;
            //            iter.first->second.file_name_ = filename;
            //            iter.first->second.buffer_type_ = buffer_type;
            //
            //            return &(iter.first->second);
            break;
        }
        case BufferType::kInvalid: {
            LOG_ERROR("Buffer type is invalid");
            break;
        }
    }
    return nullptr;
}
/**
 *
 * @param file_dir  current dir i.e. table/segment
 * @param filename
 * @param buffer_size
 * @return
 */
BufferHandle *BufferManager::AllocateBufferHandle(const SharedPtr<String> &file_dir, const SharedPtr<String> &filename, SizeT buffer_size) {
    SharedPtr<String> full_name{};
    if (file_dir.get() == nullptr or file_dir->empty()) {
        full_name = filename;
    } else {
        full_name = MakeShared<String>(Format("{}/{}", *file_dir, *filename));
    }

    UniqueLock<RWMutex> w_locker(rw_locker_);
    // call BufferHandle constructor here.
    auto iter = buffer_map_.emplace(*full_name, this);
    // `buffer_handle` here is newly constructed if `full_name` is not in buffer_map, or the existing one.
    BufferHandle *buffer_handle = &iter.first->second;
    bool success = iter.second;
    buffer_handle->id_ = next_buffer_id_++;
    buffer_handle->base_dir_ = this->base_dir_;
    buffer_handle->temp_dir_ = this->temp_dir_;
    buffer_handle->current_dir_ = file_dir; // TODO: need to set the current dir table /segment
    buffer_handle->file_name_ = filename;
    buffer_handle->buffer_type_ = BufferType::kTempFile;
    buffer_handle->status_ = success ? BufferStatus::kFreed : BufferStatus::kLoaded;
    buffer_handle->offset_ = 0;

    // need to set the buffer size
    buffer_handle->buffer_size_ = buffer_size;
    return buffer_handle;
}

BufferHandle *
BufferManager::AllocateBufferHandle(const SharedPtr<String> &file_dir, const SharedPtr<String> &filename, offset_t offset, SizeT buffer_size) {
    SharedPtr<String> full_name{};
    if (file_dir.get() == nullptr or file_dir->empty()) {
        full_name = filename;
    } else {
        full_name = MakeShared<String>(Format("{}/{}", *file_dir, *filename));
    }

    UniqueLock<RWMutex> w_locker(rw_locker_);
    // call BufferHandle constructor here.
    auto iter = buffer_map_.emplace(*full_name, this);
    // `buffer_handle` here is newly constructed if `full_name` is not in buffer_map, or the existing one.
    BufferHandle *buffer_handle = &iter.first->second;
    bool success = iter.second;
    buffer_handle->id_ = next_buffer_id_++;
    buffer_handle->base_dir_ = this->base_dir_;
    buffer_handle->temp_dir_ = this->temp_dir_;
    buffer_handle->current_dir_ = file_dir; // TODO: need to set the current dir table /segment
    buffer_handle->file_name_ = filename;
    buffer_handle->buffer_type_ = BufferType::kTempFile;
    buffer_handle->status_ = success ? BufferStatus::kFreed : BufferStatus::kLoaded;
    buffer_handle->offset_ = offset;

    // need to set the buffer size
    buffer_handle->buffer_size_ = buffer_size;
    return buffer_handle;
}

UniquePtr<String> BufferManager::Free(SizeT need_memory_size) {
    while (current_memory_size_ + need_memory_size >= mem_limit_) {
        // Need to GC
        BufferHandle *dequeued_buffer_handle;
        if (queue_.try_dequeue(dequeued_buffer_handle)) {
            if (dequeued_buffer_handle == nullptr) {
                Error<StorageException>("Null buffer handle", __FILE_NAME__, __LINE__);
            } else {
                dequeued_buffer_handle->FreeData();
            }
        } else {
            UniquePtr<String> err_msg = MakeUnique<String>("Out of memory");
            LOG_ERROR(*err_msg);
            return err_msg;
        }
    }
    return nullptr;
}

} // namespace infinity
