#include <pch.h>
#include "D_ChanceToMessUpDish.h"

D_ChanceToMessUpDish::D_ChanceToMessUpDish()
{}

void D_ChanceToMessUpDish::on_enter()
{
    BehaviorNode::on_enter();
    display_leaf_text();

    float messUp = RNG::range(0.0f, 1.0f);
    if (messUp < 0.25f)
        on_failure();
}

void D_ChanceToMessUpDish::on_update(float dt)
{
    BehaviorNode* child = children.front();
    child->tick(dt);

    if (child->succeeded())
        on_success();

    display_leaf_text();
}
