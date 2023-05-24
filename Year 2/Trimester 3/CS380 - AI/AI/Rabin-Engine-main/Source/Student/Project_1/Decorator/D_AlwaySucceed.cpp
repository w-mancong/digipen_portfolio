#include <pch.h>
#include "D_AlwaySucceed.h"

D_AlwaySucceed::D_AlwaySucceed()
{}

void D_AlwaySucceed::on_enter()
{
    BehaviorNode::on_enter();
}

void D_AlwaySucceed::on_update(float dt)
{
    BehaviorNode* child = children.front();
    child->tick(dt);

    if(child->failed() || child->succeeded() || child->is_suspended())
        on_success();
}
