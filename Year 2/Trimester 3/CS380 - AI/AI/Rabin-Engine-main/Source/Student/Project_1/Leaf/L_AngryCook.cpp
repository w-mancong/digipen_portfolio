#include <pch.h>
#include "L_AngryCook.h"

L_AngryCook::L_AngryCook() : timer{ 0.0f }, angle{ 0.0f }, flag{ -1.0f }
{}

void L_AngryCook::on_enter()
{
    BehaviorNode::on_leaf_enter();
    timer = RNG::range(2.0f, 4.5f);

    agent->set_color({1.0f, 0.0f, 0.0f});
}

void L_AngryCook::on_update(float dt)
{
    float constexpr SPEED{ 1250.0f }, MAX_ANGLE{ 15.0f };
    angle += SPEED * dt * flag;
    if (angle <= -MAX_ANGLE)
        flag = 1.0f;
    else if (angle >= MAX_ANGLE)
        flag = -1.0f;
    agent->set_roll(MyVar::DegToRad(angle));

    timer -= dt;
    if (timer < 0.0f)
        on_success();

    display_leaf_text();
}

void L_AngryCook::on_exit()
{
    agent->set_roll(0.0f);
    agent->set_color(agent->get_blackboard().get_value<Vec3>("originalColor"));
}
