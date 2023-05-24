#include <pch.h>
#include "D_Timer.h"

D_Timer::D_Timer()
{}

void D_Timer::on_enter()
{
    BehaviorNode::on_enter();
    timer = RNG::range(3.0f, 5.0f);
}

void D_Timer::on_update(float dt)
{
    timer -= dt;
    if (timer < 0.0f)
    {
        on_failure();
        return;
    }

    BehaviorNode* child = children.front();
    child->tick(dt);

    if (child->succeeded())
        on_success();
}
