#include <pch.h>
#include "L_GoToKitchen.h"

L_GoToKitchen::L_GoToKitchen()
{}

void L_GoToKitchen::on_enter()
{
    BehaviorNode::on_leaf_enter();

    targetPosition = agent->get_position();
    targetPosition.x = MyVar::CELL_SIZE * 15.0f + MyVar::HALF_CELL_SIZE;
}

void L_GoToKitchen::on_update(float dt)
{
    bool result = agent->move_toward_point(targetPosition, dt);

    if (result)
        on_success();

    display_leaf_text();
}
