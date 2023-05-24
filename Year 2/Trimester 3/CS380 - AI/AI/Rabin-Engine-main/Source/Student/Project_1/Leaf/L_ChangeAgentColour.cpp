#include <pch.h>
#include "L_ChangeAgentColour.h"
#include "Agent/BehaviorAgent.h"

void L_ChangeAgentColour::on_enter()
{
	bool shouldOrder = agent->get_blackboard().get_value<bool>("Should order");
	if(!shouldOrder)
		agent->set_color( Vec3(1.0f, 0.59f, 0.59f) );
	else
		agent->set_color(Vec3(0.74f, 1.0f, 0.66f));
	agent->get_blackboard().set_value("baseColor", agent->get_color());

	on_success();
	display_leaf_text();
}