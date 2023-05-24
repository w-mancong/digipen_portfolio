#include <pch.h>
#include "D_RunOnlyOnce.h"

D_RunOnlyOnce::D_RunOnlyOnce() : ran{ false }
{}

void D_RunOnlyOnce::on_enter()
{
    if (!ran)
    {
        ran = true;
        BehaviorNode::on_enter();
        return;
    }
    on_failure();
}

void D_RunOnlyOnce::on_update(float dt)
{
    BehaviorNode* child = children.front();
    child->tick(dt);

    if (child->succeeded())
        on_success();
}
