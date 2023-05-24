#include <pch.h>
#include "D_Inverter.h"

D_Inverter::D_Inverter()
{}

void D_Inverter::on_enter()
{
    BehaviorNode::on_enter();
}

void D_Inverter::on_update(float dt)
{
    BehaviorNode* child = children.front();
    child->tick(dt);

    if (child->succeeded())
        on_failure();
    else if (child->failed())
        on_success();
}
