/******************************************************************************/
/*!
\file		Collision.h
\author 	Wong Man Cong
\par    	email: w.mancong\@digipen.edu
\date   	11-02-2022
\brief		Contains struct and function declaration for aabb collision
			between two retangular collider

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/

#ifndef CSD1130_COLLISION_H_
#define CSD1130_COLLISION_H_

#include "AEEngine.h"

/**************************************************************************/
/*!
	AABB struct that contains the positional vector of min and max of the
	bounding box
	*/
/**************************************************************************/
struct AABB
{
	//AEVec2	c; // center
	//float  r[2]; // holds half width and half height
	
	AEVec2	min;
	AEVec2	max;
};

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
									const AABB & aabb2, const AEVec2 & vel2);


#endif // CSD1130_COLLISION_H_