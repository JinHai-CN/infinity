//
// Created by jinhai on 23-10-5.
//

module;

import stl;
import global_block_id;

export module block_index;

namespace infinity {

class BlockEntry;
class SegmentEntry;

export struct BlockIndex {
    void Insert(SegmentEntry *segment_entry, TxnTimeStamp timestamp);

    void Reserve(SizeT n);

    inline SizeT BlockCount() const { return global_blocks_.size(); }

    inline SizeT SegmentCount() const { return segments_.size(); }

    BlockEntry *GetBlockEntry(i32 segment_id, i16 block_id) const;

    Vector<SegmentEntry *> segments_;
    HashMap<i32, SegmentEntry *> segment_index_;
    HashMap<i32, HashMap<i16, BlockEntry *>> segment_block_index_;
    Vector<GlobalBlockID> global_blocks_;
};

} // namespace infinity
