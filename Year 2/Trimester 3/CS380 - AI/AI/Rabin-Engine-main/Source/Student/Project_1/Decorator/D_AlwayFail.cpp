#include <pch.h>
#include "D_AlwayFail.h"

D_AlwayFail::D_AlwayFail()
{}

void D_AlwayFail::on_enter()
{
    BehaviorNode::on_enter();
}

void D_AlwayFail::on_update(float dt)
{
    BehaviorNode* child = children.front();
    child->tick(dt);

    if(child->succeeded())
        on_failure();
}
