#include <pch.h>
#include "L_ServeCustomer.h"

L_ServeCustomer::L_ServeCustomer()
{}

void L_ServeCustomer::on_enter()
{
    BehaviorNode::on_leaf_enter();

    targetPosition = agent->get_blackboard().get_value<Vec3>("originalPosition");
    MyVar::completedDish[agent->get_blackboard().get_value<size_t>("queueIndex")] = false;
}

void L_ServeCustomer::on_update(float dt)
{
    bool result = agent->move_toward_point(targetPosition, dt);

    if (result)
    {
        MyVar::moveQ[agent->get_blackboard().get_value<size_t>("queueIndex")] = true;
        on_success();
    }

    display_leaf_text();
}
