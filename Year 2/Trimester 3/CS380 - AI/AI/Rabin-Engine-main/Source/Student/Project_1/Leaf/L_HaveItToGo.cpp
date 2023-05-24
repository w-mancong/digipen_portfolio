#include <pch.h>
#include "L_HaveItToGo.h"

L_HaveItToGo::L_HaveItToGo() : timer(0.0f)
{}

void L_HaveItToGo::on_enter()
{
    timer = RNG::range(0.25f, 0.75f);

    BehaviorNode::on_leaf_enter();
}

void L_HaveItToGo::on_update(float dt)
{
    timer -= dt;
    if (timer <= 0.0f)
        on_success();
}
