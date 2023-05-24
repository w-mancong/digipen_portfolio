#include <pch.h>
#include "L_OrderMessedUp.h"

void L_OrderMessedUp::on_enter()
{
	//BehaviorNode::on_leaf_enter();

	if (MyVar::orderMessedUp)
	{
		on_success();
		agent->set_color({ 1.0f, 0.0f, 0.0f });
	}
	else
		on_failure();

	MyVar::orderMessedUp = false;
	MyVar::takenOrderFromCurrentCustomer = false;
}
