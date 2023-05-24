#include <pch.h>
#include "L_WaitForChef.h"

L_WaitForChef::L_WaitForChef() : angle{ -90.0f }, flag{ 1.0f }
{}

void L_WaitForChef::on_enter()
{
    BehaviorNode::on_leaf_enter();
    MyVar::waitForChef[agent->get_blackboard().get_value<size_t>("queueIndex")] = true;
    timer = 1.0f;
}

void L_WaitForChef::on_update(float dt)
{
    if (MyVar::completedDish[agent->get_blackboard().get_value<size_t>("queueIndex")])
    {
        timer -= dt;
        if (timer < 0.0f)
            on_success();
    }

    float constexpr SPEED{ 350.0f }, MAX_ANGLE{ 15.0f };
    angle += SPEED * dt * flag;
    if (-90.0f - MAX_ANGLE >= angle)
        flag = 1.0f;
    else if (-90.0f + MAX_ANGLE <= angle)
        flag = -1.0f;
    agent->set_yaw(MyVar::DegToRad(angle));

    display_leaf_text();
}
