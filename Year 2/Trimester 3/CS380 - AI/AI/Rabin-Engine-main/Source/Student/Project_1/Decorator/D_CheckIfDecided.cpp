#include <pch.h>
#include "D_CheckIfDecided.h"

D_CheckIfDecided::D_CheckIfDecided()
{
}

void D_CheckIfDecided::on_enter()
{
    // this node will be used to decide if agent go into restaurant or continue walking pass the store
    BehaviorNode::on_enter();
    if (decisions[0])
        return;
    decisions[0] = true;
    // TODO: make decisions[1] become rng based;
    decisions[1] = RNG::coin_toss();
    //decisions[1] = true;
    //decisions[1] = false;
    Blackboard& bb = agent->get_blackboard();
    bb.set_value("Should order", decisions[1]);
}

void D_CheckIfDecided::on_update(float dt)
{
    BehaviorNode* child = children.front();
    child->tick(dt);

    if (!decisions[1])
        on_failure();
    else
        on_success();
}
