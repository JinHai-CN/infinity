//
// Created by jinhai on 23-4-27.
//

#include "executor/fragment/plan_fragment.h"
#include "src/scheduler/fragment_context.h"

namespace infinity{

PlanFragment::PlanFragment(QueryContext* query_context,
                           SinkType sink_type,
                           PhysicalOperator* last_phys_op) {
    AddSinkNode(query_context, sink_type, last_phys_op->GetOutputNames(), last_phys_op->GetOutputTypes());
}

Vector<UniquePtr<FragmentTask>>&
PlanFragment::CreateTasks(QueryContext* query_context) {
    context_ = FragmentContext::MakeFragmentContext(query_context, this);
    return context_->Tasks();
}

void
PlanFragment::AddSourceNode(QueryContext* query_context,
                            SourceType source_type,
                            const SharedPtr<Vector<String>>& names,
                            const SharedPtr<Vector<SharedPtr<DataType>>>& types) {
    source_ = MakeUnique<PhysicalSource>(query_context->GetNextNodeID(),
                                         source_type,
                                         names,
                                         types);
}

void
PlanFragment::AddSinkNode(QueryContext* query_context,
                          SinkType sink_type,
                          const SharedPtr<Vector<String>>& names,
                          const SharedPtr<Vector<SharedPtr<DataType>>>& types) {
    sink_ = MakeUnique<PhysicalSink>(query_context->GetNextNodeID(),
                                     sink_type,
                                     names,
                                     types);
}

SharedPtr<Vector<String>>
PlanFragment::ToString(){
    auto result = std::make_shared<Vector<String>>();
    if(source_ != nullptr) {
        result->push_back(source_->GetName());
    }
    for(auto& op: operators_){
        result->push_back(op->GetName());
    }
    if (sink_ != nullptr) {
        result->push_back(sink_->GetName());
    }
    return result;
}

SharedPtr<Table>
PlanFragment::GetResult() {
    return context_->GetResult();
}

}