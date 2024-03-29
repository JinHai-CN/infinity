//
// Created by JinHai on 2022/11/12.
//

module;

import stl;
import selection;
import infinity_assert;
import infinity_exception;
import column_vector;
import parser;
import value;
import third_party;
import default_values;
import serialize;

module data_block;

namespace infinity {

void DataBlock::Init(const DataBlock *input, const SharedPtr<Selection> &input_select) {
    Assert<StorageException>(!initialized, "Data block was initialized before.", __FILE_NAME__, __LINE__);
    Assert<StorageException>(input != nullptr && input_select.get() != nullptr, "Invalid input data block or select", __FILE_NAME__, __LINE__);

    column_count_ = input->column_count();
    Assert<StorageException>(column_count_ > 0, "Empty column vectors.", __FILE_NAME__, __LINE__);
    column_vectors.reserve(column_count_);
    for (SizeT idx = 0; idx < column_count_; ++idx) {
        column_vectors.emplace_back(MakeShared<ColumnVector>(input->column_vectors[idx]->data_type()));
        column_vectors.back()->Initialize(*(input->column_vectors[idx]), *input_select);
    }
    capacity_ = column_vectors[0]->capacity();
    initialized = true;
    this->Finalize();
}

void DataBlock::Init(const SharedPtr<DataBlock> &input, const SharedPtr<Selection> &input_select) { Init(input.get(), input_select); }

void DataBlock::Init(const SharedPtr<DataBlock> &input, SizeT start_idx, SizeT end_idx) {
    Assert<StorageException>(!initialized, "Data block was initialized before.", __FILE_NAME__, __LINE__);
    Assert<StorageException>(input.get() != nullptr, "Invalid input data block", __FILE_NAME__, __LINE__);
    column_count_ = input->column_count();
    Assert<StorageException>(column_count_ > 0, "Empty column vectors.", __FILE_NAME__, __LINE__);
    column_vectors.reserve(column_count_);
    for (SizeT idx = 0; idx < column_count_; ++idx) {
        column_vectors.emplace_back(MakeShared<ColumnVector>(input->column_vectors[idx]->data_type()));
        column_vectors.back()->Initialize(*(input->column_vectors[idx]), start_idx, end_idx);
    }
    capacity_ = column_vectors[0]->capacity();
    initialized = true;
    this->Finalize();
}

void DataBlock::Init(const Vector<SharedPtr<DataType>> &types, SizeT capacity) {
    Assert<StorageException>(!initialized, "Data block was initialized before.", __FILE_NAME__, __LINE__);
    if (types.empty()) {
        Error<StorageException>("Empty data types collection.", __FILE_NAME__, __LINE__);
    }
    column_count_ = types.size();
    column_vectors.reserve(column_count_);
    for (SizeT idx = 0; idx < column_count_; ++idx) {
        column_vectors.emplace_back(MakeShared<ColumnVector>(types[idx]));
        column_vectors[idx]->Initialize(ColumnVectorType::kFlat, capacity);
    }
    capacity_ = capacity;
    initialized = true;
}

void DataBlock::Init(const Vector<SharedPtr<ColumnVector>> &input_vectors) {
    Assert<StorageException>(!input_vectors.empty(), "Empty column vectors.", __FILE_NAME__, __LINE__);
    column_count_ = input_vectors.size();
    column_vectors = input_vectors;
    capacity_ = column_vectors[0]->capacity();
    initialized = true;
    Finalize();
}

void DataBlock::UnInit() {
    if (!initialized) {
        // Already in un-initialized state
        return;
    }

    column_vectors.clear();

    row_count_ = 0;
    initialized = false;
    finalized = false;
}

void DataBlock::Reset() {

    // Reset behavior:
    // Reset each column into just initialized status.
    // No data is appended into any column.

    for (SizeT i = 0; i < column_count_; ++i) {
        column_vectors[i]->Reset();
        column_vectors[i]->Initialize();
    }

    row_count_ = 0;
}

Value DataBlock::GetValue(SizeT column_index, SizeT row_index) const { return column_vectors[column_index]->GetValue(row_index); }

void DataBlock::SetValue(SizeT column_index, SizeT row_index, const Value &val) {
    Assert<StorageException>(column_index < column_count_,
                             Format("Attempt to access invalid column index: {}", column_index),
                             __FILE_NAME__,
                             __LINE__);
    column_vectors[column_index]->SetValue(row_index, val);
}

void DataBlock::AppendValue(SizeT column_index, const Value &value) {
    Assert<StorageException>(column_index < column_count_,
                             Format("Attempt to access invalid column index: {} in column count: {}", column_index, column_count_),
                             __FILE_NAME__,
                             __LINE__);
    column_vectors[column_index]->AppendValue(value);
    finalized = false;
}

void DataBlock::AppendValueByPtr(SizeT column_index, const ptr_t value_ptr) {
    Assert<StorageException>(column_index < column_count_,
                             Format("Attempt to access invalid column index: {} in column count: {}", column_index, column_count_),
                             __FILE_NAME__,
                             __LINE__);

    column_vectors[column_index]->AppendByPtr(value_ptr);
    finalized = false;
}

void DataBlock::Finalize() {
    bool first_flat_column_vector = false;
    SizeT row_count = 0;
    for (SizeT idx = 0; idx < column_count_; ++idx) {
        if (column_vectors[idx]->vector_type() == ColumnVectorType::kConstant) {
            continue;
        } else {
            if (first_flat_column_vector) {
                if (row_count != column_vectors[idx]->Size()) {
                    Error<StorageException>("Column vectors in same data block have different size.", __FILE_NAME__, __LINE__);
                }
            } else {
                first_flat_column_vector = true;
                row_count = column_vectors[idx]->Size();
            }
        }
    }
    row_count_ = row_count;
    finalized = true;
}

String DataBlock::ToString() const {
    StringStream ss;
    //    for (SizeT idx = 0; idx < column_count_; ++idx) {
    //        ss << "column " << idx << Endl;
    //        ss << column_vectors[idx]->ToString() << Endl;
    //    }
    return ss.str();
}

void DataBlock::FillRowIDVector(SharedPtr<Vector<RowID>> &row_ids, u32 block_id) const {
    Assert<StorageException>(finalized, "DataBlock isn't finalized.", __FILE_NAME__, __LINE__);
    for (u32 offset = 0; offset < row_count_; ++offset) {
        row_ids->emplace_back(INVALID_SEGMENT_ID, block_id, offset);
    }
}

void DataBlock::UnionWith(const SharedPtr<DataBlock> &other) {
    Assert<StorageException>(this->row_count_ == other->row_count_, "Attempt to union two block with different row count", __FILE_NAME__, __LINE__);
    Assert<StorageException>(this->capacity_ == other->capacity_, "Attempt to union two block with different row count", __FILE_NAME__, __LINE__);
    Assert<StorageException>(this->initialized && other->initialized, "Attempt to union two block with different row count", __FILE_NAME__, __LINE__);
    Assert<StorageException>(this->finalized == other->finalized, "Attempt to union two block with different row count", __FILE_NAME__, __LINE__);
    column_count_ += other->column_count_;
    column_vectors.reserve(column_count_);
    column_vectors.insert(column_vectors.end(), other->column_vectors.begin(), other->column_vectors.end());
}

void DataBlock::AppendWith(const SharedPtr<DataBlock> &other) { AppendWith(other.get()); }

void DataBlock::AppendWith(const DataBlock *other) {
    if (other->column_count() != this->column_count()) {
        Error<StorageException>(
            Format("Attempt merge block with column count {} into block with column count {}", other->column_count(), this->column_count()),
            __FILE_NAME__,
            __LINE__);
    }
    if (this->row_count_ + other->row_count_ > this->capacity_) {
        Error<StorageException>(Format("Attempt append block with row count {} into block with row count {}, "
                                       "which exceeds the capacity {}",
                                       other->row_count(),
                                       this->row_count(),
                                       this->capacity()),
                                __FILE_NAME__,
                                __LINE__);
    }

    SizeT column_count = this->column_count();
    for (SizeT idx = 0; idx < column_count; ++idx) {
        this->column_vectors[idx]->AppendWith(*other->column_vectors[idx]);
    }
}

void DataBlock::AppendWith(const SharedPtr<DataBlock> &other, SizeT from, SizeT count) {
    if (other->column_count() != this->column_count()) {
        Error<StorageException>(
            Format("Attempt merge block with column count {} into block with column count {}", other->column_count(), this->column_count()),
            __FILE_NAME__,
            __LINE__);
    }
    if (this->row_count_ + count > this->capacity_) {
        Error<StorageException>(Format("Attempt append block with row count {} into block with row count{}, "
                                       "which exceeds the capacity {}",
                                       count,
                                       this->row_count(),
                                       this->capacity()),
                                __FILE_NAME__,
                                __LINE__);
    }
    SizeT column_count = this->column_count();
    for (SizeT idx = 0; idx < column_count; ++idx) {
        this->column_vectors[idx]->AppendWith(*other->column_vectors[idx], from, count);
    }
}

bool DataBlock::operator==(const DataBlock &other) const {
    if (!this->initialized && !other.initialized)
        return true;
    if (!this->initialized || !other.initialized || this->column_count_ != other.column_count_)
        return false;
    for (int i = 0; i < this->column_count_; i++) {
        const SharedPtr<ColumnVector> &column1 = this->column_vectors[i];
        const SharedPtr<ColumnVector> &column2 = other.column_vectors[i];
        if (column1.get() == nullptr || column2.get() == nullptr || *column1 != *column2)
            return false;
    }
    return true;
}

i32 DataBlock::GetSizeInBytes() const {
    Assert<StorageException>(finalized, "Data block is not finalized.", __FILE_NAME__, __LINE__);
    i32 size = sizeof(i32);
    for (int i = 0; i < column_count_; i++) {
        size += this->column_vectors[i]->GetSizeInBytes();
    }
    return size;
}

void DataBlock::WriteAdv(char *&ptr) const {
    Assert<StorageException>(finalized, "Data block is not finalized.", __FILE_NAME__, __LINE__);
    WriteBufAdv<i32>(ptr, column_count_);
    for (int i = 0; i < column_count_; i++) {
        this->column_vectors[i]->WriteAdv(ptr);
    }
    return;
}

SharedPtr<DataBlock> DataBlock::ReadAdv(char *&ptr, i32 maxbytes) {
    char *const ptr_end = ptr + maxbytes;
    i32 column_count = ReadBufAdv<i32>(ptr);
    Vector<SharedPtr<ColumnVector>> column_vectors;
    for (int i = 0; i < column_count; i++) {
        maxbytes = ptr_end - ptr;
        Assert<StorageException>(maxbytes > 0, "ptr goes out of range when reading DataBlock", __FILE_NAME__, __LINE__);
        SharedPtr<ColumnVector> column_vector = ColumnVector::ReadAdv(ptr, maxbytes);
        column_vectors.push_back(column_vector);
    }
    SharedPtr<DataBlock> block = DataBlock::Make();
    block->Init(column_vectors);
    block->Finalize();
    maxbytes = ptr_end - ptr;
    Assert<StorageException>(maxbytes >= 0, "ptr goes out of range when reading DataBlock", __FILE_NAME__, __LINE__);
    return block;
}

} // namespace infinity
