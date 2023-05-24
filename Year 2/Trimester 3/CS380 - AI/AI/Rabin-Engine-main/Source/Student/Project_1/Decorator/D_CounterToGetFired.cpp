#include <pch.h>
#include "D_CounterToGetFired.h"

D_CounterToGetFired::D_CounterToGetFired() : counter{ 0 }
{}

void D_CounterToGetFired::on_enter()
{
    BehaviorNode::on_enter();

    if (++counter <= 2)
        on_failure();
}

void D_CounterToGetFired::on_update(float dt)
{
    BehaviorNode* child = children.front();
    child->tick(dt);

    if (child->succeeded())
    {
        counter = 0;
        on_success();
    }
    display_leaf_text();
}

