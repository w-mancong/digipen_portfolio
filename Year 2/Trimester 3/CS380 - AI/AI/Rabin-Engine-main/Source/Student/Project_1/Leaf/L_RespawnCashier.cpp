#include <pch.h>
#include "L_RespawnCashier.h"

L_RespawnCashier::L_RespawnCashier() : timer{ 0.0f }, angle{ 0.0f }, scale{ 0.0f }
{}

void L_RespawnCashier::on_enter()
{
    timer = 10.0f;
    BehaviorNode::on_leaf_enter();
}

void L_RespawnCashier::on_update(float dt)
{
    display_leaf_text();
    timer -= dt;
    if (timer > 0.0f)
        return;

    float constexpr SPEED{ 2500.0f }, SCALE_SPEED{ 1.0f };
    angle += SPEED * dt;

    agent->set_yaw( MyVar::DegToRad(angle) );
    scale += SCALE_SPEED * dt;
    if (scale >= 2.0f)
    {
        scale = 2.0f;
        on_success();
    }
    agent->set_scaling(scale);
}

void L_RespawnCashier::on_exit()
{
    agent->set_yaw(MyVar::DegToRad(-90.0f));
    MyVar::fired = false;
    MyVar::takenOrderFromCurrentCustomer = false;
    MyVar::orderMessedUp = false;
    MyVar::lowestID[LEFT_Q] = std::numeric_limits<int>::max();
    MyVar::takeOrder[LEFT_Q] = false;
}