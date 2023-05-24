#include <pch.h>
#include "L_ResetCustomerVariables.h"

L_ResetCustomerVariables::L_ResetCustomerVariables()
{}

void L_ResetCustomerVariables::on_enter()
{
    bool left = agent->get_blackboard().get_value<bool>("leftQueue");
    if (left)
    {
        MyVar::takeOrder[LEFT_Q] = false;
        //MyVar::moveQ[LEFT_Q] = true;
    }
    else
    {
        MyVar::takeOrder[RIGHT_Q] = false;
        //MyVar::moveQ[RIGHT_Q] = true;
    }
    agent->get_blackboard().set_value("ordered", true);
    on_success();
}