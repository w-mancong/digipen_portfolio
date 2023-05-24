#include <pch.h>
#include "L_LeaveStore.h"

L_LeaveStore::L_LeaveStore()
{}

void L_LeaveStore::on_enter()
{
    BehaviorNode::on_leaf_enter();
    targetPosition = Vec3(MyVar::CELL_SIZE * 10.0f + MyVar::HALF_CELL_SIZE, 0.0f, MyVar::CELL_SIZE * 10.0f);
    agent->get_blackboard().set_value("leavingStore", true);
}

void L_LeaveStore::on_update(float dt)
{
    bool result = agent->move_toward_point(targetPosition, dt);

    if (result)
    {
        ++counter;
        if (counter == 1)   // still in store, next to move towards entrance and out to the street
            targetPosition.x = -10.0f;
        else if (counter == 2)
        {   // now agent is outside of store, choose a direction to go
            bool goLeft = RNG::coin_toss();
            if (goLeft)
                targetPosition.z = -50.0f;
            else
                targetPosition.z = 170.0f;
        }
        else if (counter == 3)
        {   // out of sight, out of mind by Matthew
            on_success();
        }
    }

    display_leaf_text();
}

void L_LeaveStore::on_exit()
{
    agents->destroy_agent(agent);
    --MyVar::totalCustomers;
}
