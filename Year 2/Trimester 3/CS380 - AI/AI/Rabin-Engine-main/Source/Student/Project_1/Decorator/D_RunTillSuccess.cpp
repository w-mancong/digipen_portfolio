#include <pch.h>
#include "D_RunTillSuccess.h"

D_RunTillSuccess::D_RunTillSuccess()
{}

void D_RunTillSuccess::on_enter()
{
    BehaviorNode::on_enter();
}

void D_RunTillSuccess::on_update(float dt)
{
    BehaviorNode* child = children.front();
    child->tick(dt);

    if (child->succeeded())
    {
        on_success();
    }
    else if (child->failed())
    {
        child->set_status(NodeStatus::READY);
    }
}
