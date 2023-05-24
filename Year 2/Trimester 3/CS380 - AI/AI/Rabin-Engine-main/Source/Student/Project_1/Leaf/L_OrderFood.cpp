#include <pch.h>
#include "L_OrderFood.h"

L_OrderFood::L_OrderFood() : angle{ 90.0f }, flag{ -1.0f }
{}

void L_OrderFood::on_enter()
{
    BehaviorNode::on_leaf_enter();
    leftQueue = agent->get_blackboard().get_value<bool>("leftQueue");
}

void L_OrderFood::on_update(float dt)
{
    float constexpr SPEED{ 350.0f }, MAX_ANGLE{ 15.0f };
    angle += SPEED * dt * flag;
    if (90.0f + MAX_ANGLE <= angle)
        flag = -1.0f;
    else if (90.0f - MAX_ANGLE >= angle)
        flag = 1.0f;
    agent->set_yaw( MyVar::DegToRad(angle) );
    if (leftQueue)
    {
        if (MyVar::moveQ[LEFT_Q])
            on_success();
    }
    else
    {
        if (MyVar::moveQ[RIGHT_Q])
            on_success();
    }

    display_leaf_text();
}

void L_OrderFood::on_exit()
{
    agent->get_blackboard().set_value("ordered", true);
    MyVar::lowestID[agent->get_blackboard().get_value<size_t>("queueIndex")] = std::numeric_limits<int>::max();
}
