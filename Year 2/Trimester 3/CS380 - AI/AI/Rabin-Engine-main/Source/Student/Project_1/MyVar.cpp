#include <pch.h>
#include "MyVar.h"

float const MyVar::CELL_SIZE	  = terrain->mapSizeInWorld * 0.05f, 
			MyVar::HALF_CELL_SIZE = MyVar::CELL_SIZE * 0.5f;

bool MyVar::takeOrder[2]{ false, false }, MyVar::orderMessedUp{ false }, 
	 MyVar::takenOrderFromCurrentCustomer{ false }, MyVar::moveQ[2]{ false, false }, MyVar::fired{ false };

bool MyVar::completedDish[2]{ false, false }, MyVar::waitForChef[2]{ false, false };

int MyVar::queueID[2]{ 0, 0 }, MyVar::lowestID[2]{ 0, 0 };
int MyVar::totalCustomers{ 0 };

Table MyVar::tables[7]{};

float MyVar::DegToRad(float deg)
{
	return deg / 180.0f * PI;
}