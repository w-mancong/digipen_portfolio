#include <pch.h>
#include "L_Idle.h"

L_Idle::L_Idle() : timer(0.0f)
{}

void L_Idle::on_enter()
{
    timer = RNG::range(30.0f, 75.0f);

	BehaviorNode::on_leaf_enter();
}

void L_Idle::on_update(float dt)
{
    timer -= dt;

    if (timer < 0.0f)
        on_success();

    display_leaf_text();
}
