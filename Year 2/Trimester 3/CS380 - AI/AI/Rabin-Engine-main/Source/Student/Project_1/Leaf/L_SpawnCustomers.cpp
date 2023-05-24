#include <pch.h>
#include "L_SpawnCustomers.h"
#include "BehaviorTreeBuilder.h"

L_SpawnCustomers::L_SpawnCustomers()
{}

void L_SpawnCustomers::on_enter()
{
    BehaviorNode::on_leaf_enter();

    int spawnCustomers = RNG::range(7, 15);
    if (spawnCustomers + MyVar::totalCustomers > MyVar::MAX_CUSTOMERS)
        spawnCustomers = MyVar::MAX_CUSTOMERS - MyVar::totalCustomers;

    for (int i{}; i < spawnCustomers; ++i)
    {
        BehaviorAgent* ba = agents->create_behavior_agent("Customer", BehaviorTreeTypes::Customer);
        bool spawnLeft = RNG::coin_toss();
        float offset = RNG::range(-30.0f, 30.0f);
        if (spawnLeft)
            ba->set_position(Vec3(-10.0f, 0.0f, -40.0f + offset));
        else
            ba->set_position(Vec3(-10.0f, 0.0f, 165.0f + offset));
        ba->set_scaling(2.0f);
        ba->set_movement_speed(RNG::range(10.0f, 20.0f));
        ba->get_blackboard().set_value("ordered", false);
        ba->get_blackboard().set_value("leftQueue", false);
        ba->get_blackboard().set_value("leavingStore", false);
        ba->get_blackboard().set_value("spawnLeft", spawnLeft);
        ba->get_blackboard().set_value("queueID", std::numeric_limits<int>::max());
        ba->get_blackboard().set_value( "queueIndex", std::numeric_limits<size_t>::max() );
    }

    MyVar::totalCustomers += spawnCustomers;

    on_success();
    display_leaf_text();
}
