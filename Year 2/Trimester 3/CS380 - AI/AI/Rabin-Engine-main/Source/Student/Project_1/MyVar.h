#pragma once

struct Table
{
	bool isEmpty;
	Vec3 position;	// table's position
};

namespace Var
{
	size_t constexpr const TABLE_SIZE{ 7 };
}

class MyVar
{
public:
	static float const CELL_SIZE, HALF_CELL_SIZE;
	static bool takeOrder[2], orderMessedUp, takenOrderFromCurrentCustomer, moveQ[2], fired;
	static bool completedDish[2], waitForChef[2];
	static Table tables[Var::TABLE_SIZE];
	static int queueID[2], lowestID[2];
	static int totalCustomers;
	static int constexpr MAX_CUSTOMERS{ 35 };

	static float DegToRad(float deg);
};