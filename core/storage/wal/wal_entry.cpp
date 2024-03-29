module;

import stl;
import serialize;
import table_def;
import data_block;
import infinity_assert;
import infinity_exception;
import parser;
import third_party;
import crc;

module wal_entry;

namespace infinity {

SharedPtr<WalCmd> WalCmd::ReadAdv(char *&ptr, i32 maxbytes) {
    char *const ptr_end = ptr + maxbytes;
    SharedPtr<WalCmd> cmd = nullptr;
    WalCommandType cmd_type = ReadBufAdv<WalCommandType>(ptr);
    switch (cmd_type) {
        case WalCommandType::CREATE_DATABASE: {
            String db_name = ReadBufAdv<String>(ptr);
            cmd = MakeShared<WalCmdCreateDatabase>(db_name);
            break;
        }
        case WalCommandType::DROP_DATABASE: {
            String db_name = ReadBufAdv<String>(ptr);
            cmd = MakeShared<WalCmdDropDatabase>(db_name);
            break;
        }
        case WalCommandType::CREATE_TABLE: {
            String db_name = ReadBufAdv<String>(ptr);
            SharedPtr<TableDef> table_def = TableDef::ReadAdv(ptr, ptr_end - ptr);
            cmd = MakeShared<WalCmdCreateTable>(db_name, table_def);
            break;
        }
        case WalCommandType::DROP_TABLE: {
            String db_name = ReadBufAdv<String>(ptr);
            String table_name = ReadBufAdv<String>(ptr);
            cmd = MakeShared<WalCmdDropTable>(db_name, table_name);
            break;
        }
        case WalCommandType::IMPORT: {
            String db_name = ReadBufAdv<String>(ptr);
            String table_name = ReadBufAdv<String>(ptr);
            String segment_dir = ReadBufAdv<String>(ptr);
            cmd = MakeShared<WalCmdImport>(db_name, table_name, segment_dir);
            break;
        }
        case WalCommandType::APPEND: {
            String db_name = ReadBufAdv<String>(ptr);
            String table_name = ReadBufAdv<String>(ptr);
            SharedPtr<DataBlock> block = block->ReadAdv(ptr, ptr_end - ptr);
            cmd = MakeShared<WalCmdAppend>(db_name, table_name, block);
            break;
        }
        case WalCommandType::DELETE: {
            String db_name = ReadBufAdv<String>(ptr);
            String table_name = ReadBufAdv<String>(ptr);
            i32 cnt = ReadBufAdv<i32>(ptr);
            Vector<RowID> row_ids;
            for (i32 i = 0; i < cnt; ++i) {
                RowID row_id = ReadBufAdv<RowID>(ptr);
                row_ids.push_back(row_id);
            }
            cmd = MakeShared<WalCmdDelete>(db_name, table_name, row_ids);
            break;
        }
        case WalCommandType::CHECKPOINT: {
            i64 max_commit_ts = ReadBufAdv<i64>(ptr);
            String catalog_path = ReadBufAdv<String>(ptr);
            cmd = MakeShared<WalCmdCheckpoint>(max_commit_ts, catalog_path);
            break;
        }
        default: {
            Error<StorageException>(Format("UNIMPLEMENTED ReadAdv for WalCmd command {}", int(cmd_type)), __FILE_NAME__, __LINE__);
        }
    }
    maxbytes = ptr_end - ptr;
    Assert<StorageException>(maxbytes >= 0, "ptr goes out of range when reading WalCmd", __FILE_NAME__, __LINE__);
    return cmd;
}

bool WalCmdCreateTable::operator==(const WalCmd &other) const {
    auto other_cmd = dynamic_cast<const WalCmdCreateTable *>(&other);
    return other_cmd != nullptr && IsEqual(db_name, other_cmd->db_name) && table_def.get() != nullptr && other_cmd->table_def.get() != nullptr &&
           *table_def == *other_cmd->table_def;
}

bool WalCmdImport::operator==(const WalCmd &other) const {
    auto other_cmd = dynamic_cast<const WalCmdImport *>(&other);
    if (other_cmd == nullptr || !IsEqual(db_name, other_cmd->db_name) || !IsEqual(table_name, other_cmd->table_name) ||
        !IsEqual(segment_dir, other_cmd->segment_dir))
        return false;
    return true;
}

bool WalCmdAppend::operator==(const WalCmd &other) const {
    auto other_cmd = dynamic_cast<const WalCmdAppend *>(&other);
    if (other_cmd == nullptr || !IsEqual(db_name, other_cmd->db_name) || !IsEqual(table_name, other_cmd->table_name))
        return false;
    return true;
}

bool WalCmdDelete::operator==(const WalCmd &other) const {
    auto other_cmd = dynamic_cast<const WalCmdDelete *>(&other);
    if (other_cmd == nullptr || !IsEqual(db_name, other_cmd->db_name) || !IsEqual(table_name, other_cmd->table_name) || row_ids.size() != other_cmd->row_ids.size()) {
        return false;
    }
    for (SizeT i = 0; i < row_ids.size(); i++) {
        if (row_ids[i] != other_cmd->row_ids[i]) {
            return false;
        }
    }
    return true;
}

bool WalCmdCheckpoint::operator==(const WalCmd &other) const {
    auto other_cmd = dynamic_cast<const WalCmdCheckpoint *>(&other);
    return other_cmd != nullptr && max_commit_ts_ == other_cmd->max_commit_ts_;
}

i32 WalCmdCreateDatabase::GetSizeInBytes() const { return sizeof(WalCommandType) + sizeof(i32) + this->db_name.size(); }

i32 WalCmdDropDatabase::GetSizeInBytes() const { return sizeof(WalCommandType) + sizeof(i32) + this->db_name.size(); }

i32 WalCmdCreateTable::GetSizeInBytes() const {
    return sizeof(WalCommandType) + sizeof(i32) + this->db_name.size() + this->table_def->GetSizeInBytes();
}

i32 WalCmdDropTable::GetSizeInBytes() const {
    return sizeof(WalCommandType) + sizeof(i32) + this->db_name.size() + sizeof(i32) + this->table_name.size();
}

i32 WalCmdImport::GetSizeInBytes() const {
    return sizeof(WalCommandType) + sizeof(i32) + this->db_name.size() + sizeof(i32) + this->table_name.size() + sizeof(i32) +
           this->segment_dir.size();
}

i32 WalCmdAppend::GetSizeInBytes() const {
    return sizeof(WalCommandType) + sizeof(i32) + this->db_name.size() + sizeof(i32) + this->table_name.size() + block->GetSizeInBytes();
}

i32 WalCmdDelete::GetSizeInBytes() const {
    return sizeof(WalCommandType) + sizeof(i32) + this->db_name.size() + sizeof(i32) + this->table_name.size() + sizeof(i32) +
           row_ids.size() * sizeof(RowID);
}

i32 WalCmdCheckpoint::GetSizeInBytes() const { return sizeof(WalCommandType) + sizeof(i64) + sizeof(i32) + this->catalog_path_.size(); }

void WalCmdCreateDatabase::WriteAdv(char *&buf) const {
    WriteBufAdv(buf, WalCommandType::CREATE_DATABASE);
    WriteBufAdv(buf, this->db_name);
}

void WalCmdDropDatabase::WriteAdv(char *&buf) const {
    WriteBufAdv(buf, WalCommandType::DROP_DATABASE);
    WriteBufAdv(buf, this->db_name);
}

void WalCmdCreateTable::WriteAdv(char *&buf) const {
    WriteBufAdv(buf, WalCommandType::CREATE_TABLE);
    WriteBufAdv(buf, this->db_name);
    this->table_def->WriteAdv(buf);
}

void WalCmdDropTable::WriteAdv(char *&buf) const {
    WriteBufAdv(buf, WalCommandType::DROP_TABLE);
    WriteBufAdv(buf, this->db_name);
    WriteBufAdv(buf, this->table_name);
}

void WalCmdImport::WriteAdv(char *&buf) const {
    WriteBufAdv(buf, WalCommandType::IMPORT);
    WriteBufAdv(buf, this->db_name);
    WriteBufAdv(buf, this->table_name);
    WriteBufAdv(buf, this->segment_dir);
}

void WalCmdAppend::WriteAdv(char *&buf) const {
    WriteBufAdv(buf, WalCommandType::APPEND);
    WriteBufAdv(buf, this->db_name);
    WriteBufAdv(buf, this->table_name);
    block->WriteAdv(buf);
}

void WalCmdDelete::WriteAdv(char *&buf) const {
    WriteBufAdv(buf, WalCommandType::DELETE);
    WriteBufAdv(buf, this->db_name);
    WriteBufAdv(buf, this->table_name);
    WriteBufAdv(buf, static_cast<i32>(this->row_ids.size()));
    SizeT row_count = this->row_ids.size();
    for(SizeT idx = 0; idx < row_count; ++ idx) {
        const auto& row_id = this->row_ids[idx];
        WriteBufAdv(buf, row_id);
    }
}

void WalCmdCheckpoint::WriteAdv(char *&buf) const {
    WriteBufAdv(buf, WalCommandType::CHECKPOINT);
    WriteBufAdv(buf, this->max_commit_ts_);
    WriteBufAdv(buf, this->catalog_path_);
}

bool WalEntry::operator==(const WalEntry &other) const {
    if (this->txn_id != other.txn_id || this->commit_ts != other.commit_ts || this->cmds.size() != other.cmds.size()) {
        return false;
    }
    for (i32 i = 0; i < this->cmds.size(); i++) {
        const SharedPtr<WalCmd> &cmd1 = this->cmds[i];
        const SharedPtr<WalCmd> &cmd2 = other.cmds[i];
        if (cmd1.get() == nullptr || cmd2.get() == nullptr || (*cmd1).operator!=(*cmd2)) {
            return false;
        }
    }
    return true;
}

bool WalEntry::operator!=(const WalEntry &other) const { return !operator==(other); }

i32 WalEntry::GetSizeInBytes() const {
    i32 size = sizeof(WalEntryHeader) + sizeof(i32);
    SizeT cmd_count = cmds.size();
    for(SizeT idx = 0; idx < cmd_count; ++ idx) {
        const auto& cmd = cmds[idx];
        size += cmd->GetSizeInBytes();
    }

    size += sizeof(i32); // pad
    return size;
}

void WalEntry::WriteAdv(char *&ptr) const {
    // An entry is serialized as follows:
    // - WalEntryHeader
    // - number of WalCmd
    // - (repeated) WalCmd
    // - 4 bytes pad
    char *const saved_ptr = ptr;
    Memcpy(ptr, this, sizeof(WalEntryHeader));
    ptr += sizeof(WalEntryHeader);
    WriteBufAdv(ptr, static_cast<i32>(cmds.size()));
    SizeT cmd_count = cmds.size();
    for(SizeT idx = 0; idx < cmd_count; ++ idx) {
        const auto& cmd = cmds[idx];
        cmd->WriteAdv(ptr);
    }
    i32 size = ptr - saved_ptr + sizeof(i32);
    WriteBufAdv(ptr, size);
    WalEntryHeader *header = (WalEntryHeader *)saved_ptr;
    header->size = size;
    header->checksum = 0;
    // CRC32IEEE is equivalent to boost::crc_32_type on
    // little-endian machine.
    header->checksum = CRC32IEEE::makeCRC(reinterpret_cast<const unsigned char *>(saved_ptr), size);
    return;
}

SharedPtr<WalEntry> WalEntry::ReadAdv(char *&ptr, i32 maxbytes) {
    char *const ptr_end = ptr + maxbytes;
    Assert<StorageException>(maxbytes >= 0, "ptr goes out of range when reading WalEntry", __FILE_NAME__, __LINE__);
    SharedPtr<WalEntry> entry = MakeShared<WalEntry>();
    WalEntryHeader *header = (WalEntryHeader *)ptr;
    entry->size = header->size;
    entry->checksum = header->checksum;
    entry->txn_id = header->txn_id;
    entry->commit_ts = header->commit_ts;
    i32 size2 = *(i32 *)(ptr + entry->size - sizeof(i32));
    if (entry->size != size2) {
        return nullptr;
    }
    header->checksum = 0;
    u32 checksum2 = CRC32IEEE::makeCRC(reinterpret_cast<const unsigned char *>(ptr), entry->size);
    if (entry->checksum != checksum2) {
        return nullptr;
    }
    ptr += sizeof(WalEntryHeader);
    i32 cnt = ReadBufAdv<i32>(ptr);
    for (SizeT i = 0; i < cnt; i++) {
        maxbytes = ptr_end - ptr;
        Assert<StorageException>(maxbytes >= 0, "ptr goes out of range when reading WalEntry", __FILE_NAME__, __LINE__);
        SharedPtr<WalCmd> cmd = WalCmd::ReadAdv(ptr, maxbytes);
        entry->cmds.push_back(cmd);
    }
    ptr += sizeof(i32);
    maxbytes = ptr_end - ptr;
    Assert<StorageException>(maxbytes >= 0, "ptr goes out of range when reading WalEntry", __FILE_NAME__, __LINE__);
    return entry;
}

} // namespace infinity
