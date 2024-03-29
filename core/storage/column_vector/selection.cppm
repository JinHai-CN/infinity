//
// Created by jinhai on 23-10-17.
//
module;

import stl;
import infinity_assert;
import infinity_exception;
import global_resource_usage;
import default_values;

export module selection;

namespace infinity {

struct SelectionData {
    explicit SelectionData(SizeT count) : capacity_(count) {
        Assert<ExecutorException>(count <= u16_max, "Too large size for selection data.", __FILE_NAME__, __LINE__);
        data_ = MakeUnique<u16[]>(count);
        GlobalResourceUsage::IncrObjectCount();
    }

    ~SelectionData() { GlobalResourceUsage::DecrObjectCount(); }

    UniquePtr<u16[]> data_{};
    SizeT capacity_{};
};

export class Selection {
public:
    Selection() { GlobalResourceUsage::IncrObjectCount(); }

    ~Selection() { GlobalResourceUsage::DecrObjectCount(); }

    void Initialize(SizeT count = DEFAULT_VECTOR_SIZE) {
        storage_ = MakeShared<SelectionData>(count);
        selection_vector = storage_->data_.get();
    }

    inline void Set(SizeT selection_idx, SizeT row_idx) {
        Assert<ExecutorException>(selection_vector != nullptr, "Selection container isn't initialized", __FILE_NAME__, __LINE__);
        Assert<ExecutorException>(selection_idx < storage_->capacity_, "Exceed the selection vector capacity.", __FILE_NAME__, __LINE__);
        selection_vector[selection_idx] = row_idx;
    }

    inline void Append(SizeT row_idx) {
        Set(latest_selection_idx_, row_idx);
        ++latest_selection_idx_;
    }

    inline SizeT Get(SizeT idx) const {
        if (selection_vector == nullptr) {
            return idx;
        }
        Assert<ExecutorException>(idx < latest_selection_idx_, "Exceed the last row of the selection vector.", __FILE_NAME__, __LINE__);
        return selection_vector[idx];
    }

    inline u16 &operator[](SizeT idx) const {
        Assert<ExecutorException>(idx < latest_selection_idx_, "Exceed the last row of the selection vector.", __FILE_NAME__, __LINE__);
        return selection_vector[idx];
    }

    inline SizeT Capacity() const {
        Assert<ExecutorException>(selection_vector != nullptr, "Selection container isn't initialized", __FILE_NAME__, __LINE__);
        return storage_->capacity_;
    }

    inline SizeT Size() const {
        Assert<ExecutorException>(selection_vector != nullptr, "Selection container isn't initialized", __FILE_NAME__, __LINE__);
        return latest_selection_idx_;
    }

    void Reset() {
        storage_.reset();
        latest_selection_idx_ = 0;
        selection_vector = nullptr;
    }

private:
    SizeT latest_selection_idx_{};
    u16 *selection_vector{};
    SharedPtr<SelectionData> storage_{};
};

} // namespace infinity
