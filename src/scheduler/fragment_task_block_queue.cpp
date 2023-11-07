// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module;

//#include "blockingconcurrentqueue.h"

import fragment_task;
import stl;
import parser;

module fragment_task_block_queue;

namespace infinity {
void FragmentTaskBlockQueue::Enqueue(FragmentTask *task) {
//    queue_.enqueue(task);
}

void FragmentTaskBlockQueue::EnqueueBulk(Vector<FragmentTask *>::iterator iter, SizeT count) {
//     queue_.enqueue_bulk(iter, count);
}

void FragmentTaskBlockQueue::Dequeue(FragmentTask *&task) {
//    queue_.wait_dequeue(task);
}

} // namespace infinity