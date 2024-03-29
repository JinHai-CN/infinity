//
// Created by JinHai on 2022/11/14.
//

#pragma once

#include "parser_assert.h"
#include <cstring>
#include <string>

namespace infinity {

struct CharType {
public:
    char *ptr = nullptr;

public:
    inline explicit CharType(char *&&from_ptr) : ptr(from_ptr) { from_ptr = nullptr; }

    inline explicit CharType(size_t limit_size) { ptr = new char[limit_size]{0}; }

    inline ~CharType() {
        if (ptr != nullptr) {
            Reset();
        }
    }

    inline CharType(const CharType &other) = default;

    inline CharType(CharType &&other) noexcept {
        this->ptr = other.ptr;
        other.ptr = nullptr;
    }

    CharType &operator=(const CharType &other) {
        if (this == &other)
            return *this;
        if (ptr != nullptr) {
            Reset();
        }
        ptr = other.ptr;
        return *this;
    }

    CharType &operator=(CharType &&other) noexcept {
        if (this == &other)
            return *this;
        if (ptr != nullptr) {
            Reset();
        }
        ptr = other.ptr;
        other.ptr = nullptr;
        return *this;
    }

    bool operator==(const CharType &other) const { return strcmp(this->ptr, other.ptr); }

    inline void Initialize(const std::string &input, size_t limit) {
        ParserAssert(input.size() < limit,
                     "Attempt to store size: " + std::to_string(input.size() + 1) + " string into CharT with capacity: " + std::to_string(limit));
        memcpy(this->ptr, input.c_str(), input.size());
    }

    inline void Reset() {
        if (ptr != nullptr) {
            delete[] ptr;
            ptr = nullptr;
        }
    }

    inline void SetNull() { ptr = nullptr; }

    [[nodiscard]] inline size_t size() const { ParserAssert(this->ptr != nullptr, "Char type isn't initialized.") return std::strlen(ptr) + 1; }

    [[nodiscard]] inline char *GetDataPtr() const { return this->ptr; }

    [[nodiscard]] inline size_t Length() const { return std::strlen(ptr); }

    [[nodiscard]] std::string ToString() const { return {ptr, std::strlen(ptr)}; }
};

} // namespace infinity
