#include <pch.h>
#include "D_CheckForCustomer.h"

D_CheckForCustomer::D_CheckForCustomer()
{}

void D_CheckForCustomer::on_enter()
{
    BehaviorNode::on_enter();
    display_leaf_text();

    if (MyVar::takeOrder[agent->get_blackboard().get_value<size_t>("queueIndex")])
        on_failure();
}

void D_CheckForCustomer::on_update(float dt)
{
    BehaviorNode* child = children.front();
    child->tick(dt);

    if (child->succeeded())
        on_success();
}
