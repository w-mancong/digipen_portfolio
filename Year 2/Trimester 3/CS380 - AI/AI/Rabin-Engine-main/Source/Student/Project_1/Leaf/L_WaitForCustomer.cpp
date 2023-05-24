#include <pch.h>
#include "L_WaitForCustomer.h"

L_WaitForCustomer::L_WaitForCustomer()
{}

void L_WaitForCustomer::on_enter()
{
    BehaviorNode::on_leaf_enter();
    timer = 1.0f;
}

void L_WaitForCustomer::on_update(float dt)
{
    if (MyVar::takeOrder[agent->get_blackboard().get_value<size_t>("queueIndex")])
    {
        timer -= dt;
        if (timer < 0.0f)
            on_success();
    }
    display_leaf_text();
}
