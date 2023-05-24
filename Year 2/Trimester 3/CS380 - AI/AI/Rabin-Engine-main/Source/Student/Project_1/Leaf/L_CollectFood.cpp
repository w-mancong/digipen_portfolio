#include <pch.h>
#include "L_CollectFood.h"

L_CollectFood::L_CollectFood() : timer(0.0f)
{}

void L_CollectFood::on_enter()
{
    timer = RNG::range(0.75f, 1.25f);
    BehaviorNode::on_leaf_enter();
}

void L_CollectFood::on_update(float dt)
{
    timer -= dt;

    if (timer < 0.0f)
        on_success();

    display_leaf_text();
}
