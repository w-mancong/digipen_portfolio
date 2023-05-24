#include <pch.h>
#include "D_ChanceOfScrewUp.h"

D_ChanceOfScrewUp::D_ChanceOfScrewUp()
{}

void D_ChanceOfScrewUp::on_enter()
{
    BehaviorNode::on_enter();
    display_leaf_text();

    if (MyVar::takenOrderFromCurrentCustomer)
        return;
    float screwUp = RNG::range(0.0f, 1.0f);
    if (screwUp < 0.25f)
    {  // screws up
        on_failure();
        MyVar::orderMessedUp = true;
        MyVar::moveQ[LEFT_Q] = true;

    }
    MyVar::takenOrderFromCurrentCustomer = true;
}

void D_ChanceOfScrewUp::on_update(float dt)
{
    if (MyVar::orderMessedUp)
        return;
    BehaviorNode* child = children.front();
    child->tick(dt);

    if (child->succeeded())
        on_success();
    display_leaf_text();
}
