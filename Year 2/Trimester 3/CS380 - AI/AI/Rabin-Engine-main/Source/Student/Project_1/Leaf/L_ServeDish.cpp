#include <pch.h>
#include "L_ServeDish.h"

L_ServeDish::L_ServeDish()
{}

void L_ServeDish::on_enter()
{
    BehaviorNode::on_leaf_enter();
    targetPosition = agent->get_position();
    targetPosition.x = MyVar::CELL_SIZE * 16.0f + MyVar::HALF_CELL_SIZE;

    counter = 0;
}

void L_ServeDish::on_update(float dt)
{
    bool result = agent->move_toward_point(targetPosition, dt);

    if (result)
    {
        ++counter;
        if (counter == 1)
        {   // just reached front window, serve food
            MyVar::completedDish[agent->get_blackboard().get_value<size_t>("queueIndex")] = true;
            targetPosition = agent->get_blackboard().get_value<Vec3>("originalPosition");
        }
        else if (counter == 2)
            on_success();
    }

    display_leaf_text();
}
