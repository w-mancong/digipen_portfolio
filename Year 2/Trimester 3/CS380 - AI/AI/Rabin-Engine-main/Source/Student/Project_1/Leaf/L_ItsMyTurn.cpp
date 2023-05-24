#include <pch.h>
#include "L_ItsMyTurn.h"

L_ItsMyTurn::L_ItsMyTurn() : timer(0.0f)
{}

void L_ItsMyTurn::on_enter()
{
    timer = RNG::range(0.5f, 0.75f);
    BehaviorNode::on_leaf_enter();

    bool left = agent->get_blackboard().get_value<bool>("leftQueue");
    if (left)
        MyVar::takeOrder[LEFT_Q] = true;
    else
        MyVar::takeOrder[RIGHT_Q] = true;
}

void L_ItsMyTurn::on_update(float dt)
{
    timer -= dt;

    if (timer < 0.0f)
        on_success();
    else if (MyVar::moveQ[agent->get_blackboard().get_value<size_t>("queueIndex")])
        on_failure();

    display_leaf_text();
}
