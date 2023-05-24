#include <pch.h>
#include "L_WalkAcrossStore.h"
#include "Agent/BehaviorAgent.h"

void L_WalkAcrossStore::on_enter()
{
	on_leaf_enter();
	targetPosition = agent->get_position();
	bool spawnLeft = agent->get_blackboard().get_value<bool>("spawnLeft");
	if (spawnLeft)
		targetPosition.z = 170.0f;
	else
		targetPosition.z = -45.0f;

	delay = RNG::range(0.75f, 1.25f);
}

void L_WalkAcrossStore::on_update(float dt)
{
	display_leaf_text();

	timer += dt;
	if (timer < delay)
		return;

	if (agent->get_blackboard().get_value<bool>("Should order"))
	{
		on_failure();
		return;
	}

	bool result = agent->move_toward_point(targetPosition, dt);

	if (result)
		on_success();
}

void L_WalkAcrossStore::on_exit()
{
	if (!agent->get_blackboard().get_value<bool>("Should order"))
		agents->destroy_agent(agent);
}
