#include <pch.h>
#include "L_GetInLine.h"
#include "Agent/BehaviorAgent.h"

namespace
{
	float const CELL_SIZE = terrain->mapSizeInWorld * 0.05f,
		CELL_HALF_SIZE = CELL_SIZE * 0.5f;
}

void L_GetInLine::on_enter()
{
	on_leaf_enter();

	bool left = RNG::coin_toss();
	if (left && !MyVar::fired)
	{
		targetPosition = agent->get_position();
		targetPosition.z = CELL_SIZE * 3.0f;
		//TODO: Give agent a queue index
		agent->get_blackboard().set_value("queueID", MyVar::queueID[LEFT_Q]++);
		agent->get_blackboard().set_value("queueIndex", LEFT_Q);
	}
	else
	{
		targetPosition = agent->get_position();
		targetPosition.z = CELL_SIZE * 7.0f;
		agent->get_blackboard().set_value("queueID", MyVar::queueID[RIGHT_Q]++);
		agent->get_blackboard().set_value("queueIndex", RIGHT_Q);
	}

	agent->get_blackboard().set_value("leftQueue", left);
	delay = RNG::range(0.5f, 0.75f);
}

void L_GetInLine::on_update(float dt)
{
	timer += dt;
	if (timer < delay)
		return;

	bool result = agent->move_toward_point(targetPosition, dt);

	if (result)
		on_success();

	display_leaf_text();
}
