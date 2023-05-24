#include <pch.h>
#include "L_MoveTowardsCashier.h"
#include "Agent/BehaviorAgent.h"

void L_MoveTowardsCashier::on_enter()
{
	BehaviorNode::on_leaf_enter();

	actualPosition = targetPosition = agent->get_position();
	actualPosition.x = MyVar::CELL_SIZE * 10.0f + MyVar::HALF_CELL_SIZE;
}

void L_MoveTowardsCashier::on_update(float dt)
{
	int queueID = agent->get_blackboard().get_value<int>("queueID");
	int lowestID = MyVar::lowestID[agent->get_blackboard().get_value<size_t>("queueIndex")];
	int offset = queueID - lowestID;
	targetPosition.x = (MyVar::CELL_SIZE * 10.0f + MyVar::HALF_CELL_SIZE) - offset * (MyVar::CELL_SIZE);

	bool result = agent->move_toward_point(targetPosition, dt);
	if (result)
	{
		if(targetPosition == actualPosition)
			on_success();
	}
	display_leaf_text();
}
