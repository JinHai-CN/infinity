//
// Created by jinhai on 23-5-17.
//

module;

import stl;
import file_system;
import file_system_type;

module local_file_system;

namespace infinity {

LocalFileHandler::~LocalFileHandler() {
#if 0
    if (fd_ != -1) {
        int ret = close(fd_);
        if (ret != 0) {
            assert(0);
            // StorageError(fmt::format("Close file failed: {}", strerror(errno)));
        }
        fd_ = -1;
    }
#endif
}

UniquePtr<FileHandler> LocalFileSystem::OpenFile(const String &path, u8 flags, FileLockType lock_type) {
//    i32 file_flags{O_RDWR};
//    bool read_flag = flags & FileFlags::READ_FLAG;
//    bool write_flag = flags & FileFlags::WRITE_FLAG;
//    if (read_flag && write_flag) {
//        file_flags = O_RDWR;
//    } else if (read_flag) {
//        file_flags = O_RDONLY;
//    } else if (write_flag) {
//        file_flags = O_WRONLY;
//    } else {
//        StorageError("Please specify a read or write flag on file open");
//    }
//
//    if (write_flag) {
//        file_flags |= O_CLOEXEC; // Close fd when fork the process
//        if (flags & FileFlags::CREATE_FLAG) {
//            file_flags |= O_CREAT;
//        } else if (flags & FileFlags::TRUNCATE_CREATE) {
//            file_flags |= O_CREAT | O_TRUNC;
//        }
//
//        if (flags & FileFlags::APPEND_FLAG) {
//            file_flags |= O_APPEND;
//        }
//#if defined(__linux__)
//        if (flags & FileFlags::DIRECT_IO) {
//            file_flags |= O_DIRECT | O_SYNC;
//        }
//#endif
//    }
//
//    i32 fd = open(path.c_str(), file_flags, 0666);
//    if (fd == -1) {
//        StorageError(fmt::format("Can't open file: {}: {}", path, strerror(errno)));
//    }
//
//    if (lock_type != FileLockType::kNoLock) {
//        struct flock file_lock {};
//        memset(&file_lock, 0, sizeof(file_lock));
//        if (lock_type == FileLockType::kReadLock) {
//            file_lock.l_type = F_RDLCK;
//        } else {
//            file_lock.l_type = F_WRLCK;
//        }
//        file_lock.l_whence = SEEK_SET;
//        file_lock.l_start = 0;
//        file_lock.l_len = 0;
//        if (fcntl(fd, F_SETLK, &file_lock) == -1) {
//            StorageError(fmt::format("Can't lock file: {}, {}", path, strerror(errno)));
//        }
//    }
//    return MakeUnique<LocalFileHandler>(*this, path, fd);
}

void LocalFileSystem::Close(FileHandler &file_handler) {
#if 0
    auto local_file_handler = dynamic_cast<LocalFileHandler *>(&file_handler);
    i32 fd = local_file_handler->fd_;
    // set fd to -1 so that destructor of `LocalFileHandler` will not close the file.
    local_file_handler->fd_ = -1;
    if (close(fd) != 0) {
        StorageError(fmt::format("Can't close file: {}, {}", file_handler.path_.string(), strerror(errno)));
    }
#endif
}

i64 LocalFileSystem::Read(FileHandler &file_handler, void *data, u64 nbytes) {
#if 0
    i32 fd = ((LocalFileHandler &)file_handler).fd_;
    i64 read_count = read(fd, data, nbytes);
    if (read_count == -1) {
        StorageError(fmt::format("Can't read file: {}: {}", file_handler.path_.string(), strerror(errno)));
    }
    return read_count;
#endif
}

i64 LocalFileSystem::Write(FileHandler &file_handler, void *data, u64 nbytes) {
#if 0
    i32 fd = ((LocalFileHandler &)file_handler).fd_;
    i64 write_count = write(fd, data, nbytes);
    if (write_count == -1) {
        StorageError(fmt::format("Can't write file: {}: {}", file_handler.path_.string(), strerror(errno)));
    }
    return write_count;
#endif
}

void LocalFileSystem::Seek(FileHandler &file_handler, i64 pos) {
#if 0
    i32 fd = ((LocalFileHandler &)file_handler).fd_;
    if (0 != lseek(fd, pos, SEEK_SET)) {
        StorageError(fmt::format("Can't seek file: {}: {}: {}", file_handler.path_.string(), pos, strerror(errno)));
    }
#endif
}

SizeT LocalFileSystem::GetFileSize(FileHandler &file_handler) {
#if 0
    i32 fd = ((LocalFileHandler &)file_handler).fd_;
    struct stat s {};
    if (fstat(fd, &s) == -1) {
        return -1;
    }
    return s.st_size;
#endif
}

void LocalFileSystem::DeleteFile(const String &file_name) {
#if 0
    std::error_code error_code;
    std::filesystem::path p{file_name};
    bool is_deleted = std::filesystem::remove(p, error_code);
    if (error_code.value() == 0) {
        if (!is_deleted) {
            StorageError(fmt::format("Can't delete file: {}, {}", file_name, strerror(errno)));
        }
    } else {
        StorageError(fmt::format("Delete file {} exception: {}", file_name, error_code.message()))
    }
#endif
}

void LocalFileSystem::SyncFile(FileHandler &file_handler) {
#if 0
    i32 fd = ((LocalFileHandler &)file_handler).fd_;
    if (fsync(fd) != 0) {
        StorageError(fmt::format("fsync failed: {}, {}", file_handler.path_.string(), strerror(errno)));
    }
#endif
}

// Directory related methods
bool LocalFileSystem::Exists(const String &path) {
#if 0
    std::error_code error_code;
    std::filesystem::path p{path};
    bool is_exists = std::filesystem::exists(p, error_code);
    if (error_code.value() == 0) {
        return is_exists;
    } else {
        StorageError(fmt::format("{} exists exception: {}", path, error_code.message()))
    }
#endif
}

bool LocalFileSystem::CreateDirectoryNoExp(const String &path) {
#if 0
    std::error_code error_code;
    std::filesystem::path p{path};
    return std::filesystem::create_directories(p, error_code);
#endif
}

void LocalFileSystem::CreateDirectory(const String &path) {
#if 0
    std::error_code error_code;
    std::filesystem::path p{path};
    bool is_success = std::filesystem::create_directories(p, error_code);
    if (error_code.value() != 0) {
        StorageError(fmt::format("{} create exception: {}", path, error_code.message()))
    }
#endif
}

void LocalFileSystem::DeleteDirectory(const String &path) {
#if 0
    std::error_code error_code;
    std::filesystem::path p{path};
    bool is_deleted = std::filesystem::remove(p, error_code);
    if (error_code.value() == 0) {
        if (!is_deleted) {
            StorageError(fmt::format("Can't delete directory: {}, {}", path, strerror(errno)));
        }
    } else {
        StorageError(fmt::format("Delete directory {} exception: {}", path, error_code.message()))
    }
#endif
}

Vector<SharedPtr<DirEntry>> LocalFileSystem::ListDirectory(const String &path) {
#if 0
    std::filesystem::path dir_path(path);
    if (!is_directory(dir_path)) {
        StorageError(fmt::format("{} isn't a directory", path));
    }

    Vector<SharedPtr<DirEntry>> file_array;
    std::ranges::for_each(std::filesystem::directory_iterator{path},
                          [&](const auto &dir_entry) { file_array.emplace_back(MakeShared<DirEntry>(dir_entry)); });
    return file_array;
#endif
}

} // namespace infinity