#include <pch.h>
#include "L_WalkIntoStore.h"
#include "Agent/BehaviorAgent.h"

namespace
{
	float const CELL_SIZE = terrain->mapSizeInWorld * 0.05f,
		CELL_HALF_SIZE = CELL_SIZE * 0.5f;
}

void L_WalkIntoStore::on_enter()
{
	on_leaf_enter();

	targetPosition[0] = agent->get_position();
	targetPosition[0].z = CELL_SIZE * 10.0f;
	index = 0;

	delay = RNG::range(0.75f, 1.25f);
}

void L_WalkIntoStore::on_update(float dt)
{
	timer += dt;
	if (timer < delay)
		return;

	bool result = agent->move_toward_point(targetPosition[index], dt);

	if (result)
	{
		++index;
		targetPosition[1] = agent->get_position();
		targetPosition[1].x = CELL_SIZE + CELL_HALF_SIZE;
		timer = 0.0f;
		delay = RNG::range(0.25f, 0.75f);
		if (index >= 2)
			on_success();
	}

	display_leaf_text();
}
