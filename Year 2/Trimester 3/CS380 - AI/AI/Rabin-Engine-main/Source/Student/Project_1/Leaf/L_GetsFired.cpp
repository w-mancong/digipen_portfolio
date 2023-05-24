#include <pch.h>
#include "L_GetsFired.h"
#include "BehaviorTreeBuilder.h"

L_GetsFired::L_GetsFired() : angle{ 0.0f }, scale{ 2.0f }
{}

void L_GetsFired::on_enter()
{
    BehaviorNode::on_leaf_enter();
    scale = agent->get_scaling().x;
    MyVar::fired = true;
}

void L_GetsFired::on_update(float dt)
{
    float constexpr SPEED{ 2500.0f }, SCALE_SPEED{ 1.0f };
    angle += SPEED * dt;

    agent->set_yaw( MyVar::DegToRad(angle) );
    scale -= SCALE_SPEED * dt;
    agent->set_scaling(scale);
    if (scale <= 0.0f)
        on_success();

    display_leaf_text();
}

void L_GetsFired::on_exit()
{
    std::vector<Agent*> v = agents->get_all_agents_by_type("Customer");
    for (size_t i{}; i < v.size(); ++i)
    {
        BehaviorAgent* ba = dynamic_cast<BehaviorAgent*>(v[i]);
        if (!ba) continue;

        if(ba->get_blackboard().get_value<bool>("leftQueue") && !ba->get_blackboard().get_value<bool>("leavingStore"))
            treeBuilder->build_tree(BehaviorTreeTypes::LeaveStoreAngrily, ba);
    }
    MyVar::lowestID[LEFT_Q] = std::numeric_limits<int>::max();
}
