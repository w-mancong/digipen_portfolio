#include <pch.h>
#include "L_EmptyTable.h"

L_EmptyTable::L_EmptyTable()
{}

void L_EmptyTable::on_enter()
{
	bool gotTable = false;
	for (size_t i{}; i < Var::TABLE_SIZE; ++i)
	{
		if (!(MyVar::tables + i)->isEmpty)
			continue;
		gotTable = true;
		(MyVar::tables + i)->isEmpty = false;
		
		Blackboard& bb = agent->get_blackboard();
		bb.set_value("tableIndex", i);
		bb.set_value("tablePosition", (MyVar::tables + i)->position);
		break;
	}

	if (gotTable)
		on_success();
	else
		on_failure();

	display_leaf_text();
}
