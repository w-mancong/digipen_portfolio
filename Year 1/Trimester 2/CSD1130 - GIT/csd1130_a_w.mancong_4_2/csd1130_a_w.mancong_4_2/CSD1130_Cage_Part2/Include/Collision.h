/*!*****************************************************************************
\file		Collision.h
\author 	Wong Man Cong
\par    	DP email: w.mancong\@digipen.edu
\date   	26-03-2022
\brief
This file contains function declaration to build a line segment, collision detection
between a ball and line segment, and does response when the ball collides with
the line

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*******************************************************************************/

#ifndef CSD1130_COLLISION_H_
#define CSD1130_COLLISION_H_
#include "Matrix3x3.h"
#include "Vector2D.h"

/*!*****************************************************************************
\brief
	Struct containing relevant data members to store p0, p1 and normal of a
	line segment
*******************************************************************************/
struct LineSegment
{
	CSD1130::Vec2	m_pt0;
	CSD1130::Vec2	m_pt1;
	CSD1130::Vec2	m_normal;															//Outward normalized normal
};

/*!*****************************************************************************
\brief
	Build a line segment by forming p1 - p0 based on it's position and
	initialize this line segment's normal
\param [out] lineSegment:
	Storing information of the line segment here
\param [in] pos:
	Main position of the line (At the center)
\param [in] scale:
	Length of the line
\param [in] dir:
	Angle of the line segment (in Radians)
*******************************************************************************/
void BuildLineSegment(LineSegment &lineSegment,											//Line segment reference - input/output
					  const CSD1130::Vec2 &pos,											//Position - input
					  float scale,														//Scale - input
					  float dir);														//Direction - input

/*!*****************************************************************************
\brief
	Strut containing the relevant data members to store the position and radius
	of a circle
*******************************************************************************/
struct Circle
{
	CSD1130::Vec2	m_center;
	float			m_radius;
};

/*!*****************************************************************************
\brief
	Check for collision between a circle and line segment
\param [in] circle:
	Data containing the circle to check for collision with
\param [in] ptEnd:
	Final position of the circle in this frame to check if it's colliding with
	the line
\param [in] lineSeg:
	Data containing the line segment to check for collision with
\param [out] interPt:
	Vector2D used to store the intersection point of the circle and line segment
\param [out] normalAtCollision:
	Vector2D used to store the normal vector at the point of intersection
	between the circle and line segment
\param [out] interTime:
	Time of the intersection
\return
	True if there is a collision between the circle and line segment, else false
*******************************************************************************/
int CollisionIntersection_CircleLineSegment(const Circle &circle,						//Circle data - input
											const CSD1130::Vec2 &ptEnd,					//End circle position - input
											const LineSegment &lineSeg,					//Line segment - input
											CSD1130::Vec2 &interPt,						//Intersection point - output
											CSD1130::Vec2 &normalAtCollision,			//Normal vector at collision time - output
											float &interTime);							//Intersection time ti - output

/*!*****************************************************************************
\brief
	Calculate the new end position and the reflected vector of the circle
\param [in] ptInter:
	The point of intersection between the circle and line segment
\param [in] normal:
	Normal vector at the point of intersection between the circle and line segment
\param [in,out] ptEnd:
	Used to calculate the penetration and to store the new position after
\param [out] reflected:
	Vector2D used to store the reflected vector
*******************************************************************************/
void CollisionResponse_CircleLineSegment(const CSD1130::Vec2 &ptInter,					//Intersection position of the circle - input
										 const CSD1130::Vec2 &normal,					//Normal vector of reflection on collision time - input
										 CSD1130::Vec2 &ptEnd,							//Final position of the circle after reflection - output
										 CSD1130::Vec2 &reflected);						//Normalized reflection vector direction - output

#endif // CSD1130_COLLISION_H_