#include <pch.h>
#include "L_EatAtTable.h"

L_EatAtTable::L_EatAtTable() : angle{ 0.0f }, flag{ 1.0f }
{}

void L_EatAtTable::on_enter()
{
    BehaviorNode::on_leaf_enter();
    eatTimer = RNG::range(7.5f, 15.0f);
}

void L_EatAtTable::on_update(float dt)
{
    float constexpr SPEED{ 350.0f }, MAX_ANGLE{ 15.0f };
    angle += SPEED * dt * flag;
    if (angle >= MAX_ANGLE)
        flag = -1.0f;
    else if (angle <= 0.0f)
        flag = 1.0f;
    agent->set_pitch( MyVar::DegToRad(angle) );

    eatTimer -= dt;
    if (eatTimer <= 0.0f)
    {
        size_t index = agent->get_blackboard().get_value<size_t>("tableIndex");
        (MyVar::tables + index)->isEmpty = true;
        on_success();
    }

    display_leaf_text();
}
