#include <pch.h>
#include "L_RandomPointOutsideStore.h"
#include "Agent/BehaviorAgent.h"

void L_RandomPointOutsideStore::on_enter()
{
	BehaviorNode::on_leaf_enter();

	float const CELL_SIZE{ terrain->mapSizeInWorld / 20.0f },
				HALF_CELL_SIZE{ CELL_SIZE * 0.5f };

	targetPosition = agent->get_position();
	if (targetPosition.z < 0.0f)
		targetPosition.z = RNG::range(0.0f, CELL_SIZE * 8.0f - HALF_CELL_SIZE);
	else
		targetPosition.z = RNG::range(CELL_SIZE * 11.0f + HALF_CELL_SIZE, terrain->mapSizeInWorld);
}

void L_RandomPointOutsideStore::on_update(float dt)
{
	bool result = agent->move_toward_point(targetPosition, dt);

	if (result)
		on_success();

	display_leaf_text();
}
