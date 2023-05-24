#include <pch.h>
#include "L_AngryCustomer.h"

L_AngryCustomer::L_AngryCustomer()
{}

void L_AngryCustomer::on_enter()
{
    BehaviorNode::on_leaf_enter();
    agent->set_color({1.0f, 0.0f, 0.0f});
    on_success();
}
