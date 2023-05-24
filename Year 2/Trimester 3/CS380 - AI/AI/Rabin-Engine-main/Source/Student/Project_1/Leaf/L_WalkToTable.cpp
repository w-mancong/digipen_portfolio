#include <pch.h>
#include "L_WalkToTable.h"
#include "Agent/BehaviorAgent.h"

void L_WalkToTable::on_enter()
{
	counter = 0;
	on_leaf_enter();

	targetPosition = agent->get_position();
	targetPosition.z = MyVar::CELL_SIZE * 10.0f;
}

void L_WalkToTable::on_update(float dt)
{
	bool result = agent->move_toward_point(targetPosition, dt);

	if (result)
	{
		if (counter < 1)
			targetPosition = agent->get_blackboard().get_value<Vec3>("tablePosition");
		else
			on_success();
		++counter;
	}

	display_leaf_text();
}
