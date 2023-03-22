/******************************************************************************/
/*!
\file		Collision.cpp
\author 	Wong Man Cong
\par    	email: w.mancong\@digipen.edu
\date   	07-03-2022
\brief		This file does aabb collision check between two objects

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/

#include "main.h"

/*!**************************************************************************
\brief
	Helper function to check both static and dynamic collision between two 
	rectangular collider

\param [in] aabb1
	Bounding box of first collider
\param [in] vel1
	Velocity of first collider
\param [in] aabb2 
	Bounding box of second collider
\param [in] vel2
	Velocity of second collider

\return 
	Returns true if bounding box in this current frame is overlapping
	OR if in this current frame based on the relative velocity of the 
	two objects is going to collide
***************************************************************************/
bool CollisionIntersection_RectRect(const AABB & aabb1, const AEVec2 & vel1, 
									const AABB & aabb2, const AEVec2 & vel2)
{
	if (aabb1.min.x < aabb2.max.x && aabb1.max.x > aabb2.min.x && aabb1.min.y < aabb2.max.y && aabb1.max.y > aabb2.min.y)
		return true;
	else if (aabb1.min.x > aabb2.max.x || aabb1.max.x < aabb2.min.x || aabb1.min.y > aabb2.max.y || aabb1.max.y < aabb2.min.y)
		return false;

	AEVec2 vRel;
	vRel.x = vel2.x - vel1.x;
	vRel.y = vel2.y - vel1.y;
	float dFirst = 0.0f, dLast = 0.0f;
	float tFirst = 0.0f, tLast = g_dt;

	/******************************************************************************************************************
													X - AXIS
	*****************************************************************************************************************/
	if (0.0f > vRel.x)
	{
		// case 1
		if (aabb1.min.x > aabb2.max.x)
			return false;
		// case 4.1
		if (aabb1.max.x < aabb2.min.x)
		{
			dFirst = (aabb1.max.x - aabb2.min.x) / vRel.x;
			tFirst = dFirst > tFirst ? dFirst : tFirst;
		}
		// case 4.2
		else if (aabb1.min.x < aabb2.max.x)
		{
			dLast = (aabb1.min.x - aabb2.max.x) / vRel.x;
			tLast = dLast < tLast ? dLast : tLast;
		}
	}

	if (0.0f < vRel.x)
	{
		// case 3
		if (aabb1.max.x < aabb2.min.x)
			return false;
		// case 2.1
		if (aabb1.min.x > aabb2.max.x)
		{
			dFirst = (aabb1.min.x - aabb2.max.x) / vRel.x;
			tFirst = dFirst > tFirst ? dFirst : tFirst;
		}
		// case 2.2
		else if (aabb1.max.x > aabb2.min.x)
		{
			dLast = (aabb1.max.x - aabb2.min.x) / vRel.x;
			tLast = dLast < tLast ? dLast : tLast;
		}
	}

	/******************************************************************************************************************
													Y - AXIS
	*****************************************************************************************************************/
	if (0.0f > vRel.y)
	{
		// case 1
		if (aabb1.min.y > aabb2.max.y)
			return false;
		// case 4.1
		if (aabb1.max.y < aabb2.min.y)
		{
			dFirst = (aabb1.max.y - aabb2.min.y) / vRel.y;
			tFirst = dFirst > tFirst ? dFirst : tFirst;
		}
		// case 4.2
		else if (aabb1.min.y < aabb2.max.y)
		{
			dLast = (aabb1.min.y - aabb2.max.y) / vRel.y;
			tLast = dLast < tLast ? dLast : tLast;
		}
	}

	if (0.0f < vRel.y)
	{
		// case 3
		if (aabb1.max.y < aabb2.min.y)
			return false;
		// case 2.1
		if (aabb1.min.y > aabb2.max.y)
		{
			dFirst = (aabb1.min.y - aabb2.max.y) / vRel.y;
			tFirst = dFirst > tFirst ? dFirst : tFirst;
		}
		// case 2.2
		else if (aabb1.max.y > aabb2.min.y)
		{
			dLast = (aabb1.max.y - aabb2.min.y) / vRel.y;
			tLast = dLast < tLast ? dLast : tLast;
		}
	}

	if (tFirst > tLast)
		return false;
	return true;
}