#include <pch.h>
#include "L_CheckOnCashier.h"

L_CheckOnCashier::L_CheckOnCashier() : timer(0.0f)
{}

void L_CheckOnCashier::on_enter()
{
    BehaviorNode::on_leaf_enter();
    timer = RNG::range(1.0f, 2.0f);
    counter = 0;

    targetPosition = agent->get_position();
    targetPosition.z = MyVar::CELL_SIZE * 4.0f + MyVar::HALF_CELL_SIZE;
    interactWithCashier = false;
    doneInteraction = false;

    flag = 1.0f;
    angle = 0.0f;
}

void L_CheckOnCashier::on_update(float dt)
{
    display_leaf_text();
    bool result = agent->move_toward_point(targetPosition, dt);

    if (result)
    {
        ++counter;
        if (counter == 1)
        {   // Just reach front of cashier
            interactWithCashier = true;
        }
        // completed interactions with cashier
        else if (counter == 2 && doneInteraction)
            on_success();
    }

    if (!interactWithCashier)
        return;

    float constexpr SPEED{ 50.0f }, MAX_ANGLE{ 10.0f };
    angle += SPEED * dt * flag;
    if (angle >= MAX_ANGLE)
        flag = -1.0f;
    else if (angle <= 0.0f)
        flag = 1.0f;
    agent->set_pitch(MyVar::DegToRad(angle));

    timer -= dt;
    if (timer > 0.0f)
        return;

    counter = 1;
    targetPosition = agent->get_blackboard().get_value<Vec3>("originalPosition");
    doneInteraction = true;
}

void L_CheckOnCashier::on_exit()
{
    agent->set_pitch(0.0f);
    agent->set_yaw(MyVar::DegToRad(-90.0f));
}
