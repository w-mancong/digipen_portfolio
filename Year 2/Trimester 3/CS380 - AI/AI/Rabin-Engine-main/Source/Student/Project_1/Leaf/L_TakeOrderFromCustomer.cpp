#include <pch.h>
#include "L_TakeOrderFromCustomer.h"

L_TakeOrderFromCustomer::L_TakeOrderFromCustomer() : angle{ 0.0f }, flag{ 1.0f }
{}

void L_TakeOrderFromCustomer::on_enter()
{
    BehaviorNode::on_leaf_enter();
    timer = RNG::range(1.0f, 2.0f);
}

void L_TakeOrderFromCustomer::on_update(float dt)
{
    float constexpr SPEED{ 50.0f }, MAX_ANGLE{ 10.0f };
    angle += SPEED * dt * flag;
    if (angle >= MAX_ANGLE)
        flag = -1.0f;
    else if (angle <= 0.0f)
        flag = 1.0f;
    agent->set_pitch(MyVar::DegToRad(angle));

    timer -= dt;
    if (timer <= 0.0f)
        on_success();

    display_leaf_text();
}

void L_TakeOrderFromCustomer::on_exit()
{
    agent->set_pitch(0.0f);
}
