#include <pch.h>
#include "L_WaitForOrder.h"

L_WaitForOrder::L_WaitForOrder()
{}

void L_WaitForOrder::on_enter()
{
    BehaviorNode::on_leaf_enter();
    timer = 1.0f;
}

void L_WaitForOrder::on_update(float dt)
{
    if (MyVar::waitForChef[agent->get_blackboard().get_value<size_t>("queueIndex")])
    {
        timer -= dt;
        if(timer < 0.0f)
            on_success();
    }

    display_leaf_text();
}

void L_WaitForOrder::on_exit()
{
    MyVar::waitForChef[agent->get_blackboard().get_value<size_t>("queueIndex")] = false;
}
