#include <pch.h>
#include "L_CookDish.h"

L_CookDish::L_CookDish() : timer{0.0f}, angle{0.0f}
{}

void L_CookDish::on_enter()
{
    BehaviorNode::on_leaf_enter();
    timer = RNG::range(3.0f, 5.0f);
}

void L_CookDish::on_update(float dt)
{
    float constexpr SPEED{ 1250.0f };
    angle += SPEED * dt;
    agent->set_yaw(MyVar::DegToRad(angle));

    timer -= dt;

    if (timer < 0.0f)
        on_success();

    display_leaf_text();
}

void L_CookDish::on_exit()
{
    agent->set_yaw(MyVar::DegToRad(90.0f));
}
