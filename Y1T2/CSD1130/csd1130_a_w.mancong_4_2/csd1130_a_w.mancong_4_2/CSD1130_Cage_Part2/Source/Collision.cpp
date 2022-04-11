/*!*****************************************************************************
\file		Collision.cpp
\author 	Wong Man Cong
\par    	DP email: w.mancong\@digipen.edu
\date   	26-03-2022
\brief
This file contains definition function to build a line segment, collision detection 
between a ball and line segment, and does response when the ball collides with
the line

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*******************************************************************************/

#include "main.h"

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
void BuildLineSegment(LineSegment &lineSegment, 
					  const CSD1130::Vec2 &pos, 
					  float scale, 
					  float dir)
{
	CSD1130::Vector2D v{ 1.0f, 0.0f };								// just a random unit vector facing ->
	CSD1130::Mtx33 rot; CSD1130::Mtx33RotRad(rot, dir);
	v = (rot * v) * scale * 0.5f;
	lineSegment.m_pt0 = pos - v, lineSegment.m_pt1 = pos + v;		// init p0 and p1
	v = lineSegment.m_pt1 - lineSegment.m_pt0;						// reusing v to calculate vector form by p1 - p0
	lineSegment.m_normal.x = v.y, lineSegment.m_normal.y = -v.x;	// initialising outward normal
	CSD1130::Vector2DNormalize(lineSegment.m_normal, lineSegment.m_normal);
}

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
int CollisionIntersection_CircleLineSegment(const Circle &circle, 
											const CSD1130::Vec2 &ptEnd,
											const LineSegment &lineSeg, 
											CSD1130::Vec2 &interPt, 
											CSD1130::Vec2 &normalAtCollision,
											float &interTime)
{
	float const LNS = CSD1130::Vector2DDotProduct(lineSeg.m_normal, circle.m_center - lineSeg.m_pt0);
	if (LNS <= -circle.m_radius)
	{
		CSD1130::Vector2D const p0 = lineSeg.m_pt0 - circle.m_radius * lineSeg.m_normal, p1 = lineSeg.m_pt1 - circle.m_radius * lineSeg.m_normal;
		CSD1130::Vector2D const vel = ptEnd - circle.m_center;					// finding velocity vector
		CSD1130::Vector2D M{ vel.y, -vel.x }; CSD1130::Vector2DNormalize(M, M);	// initialize and normalize M
		if (CSD1130::Vector2DDotProduct(M, p0 - circle.m_center) * CSD1130::Vector2DDotProduct(M, p1 - circle.m_center) < 0.0f)
		{
			interTime = (CSD1130::Vector2DDotProduct(lineSeg.m_normal, lineSeg.m_pt0 - circle.m_center) - circle.m_radius) / CSD1130::Vector2DDotProduct(lineSeg.m_normal, vel);
			if (0.0f <= interTime && 1.0f >= interTime)
			{
				interPt = circle.m_center + vel * interTime;
				normalAtCollision = -lineSeg.m_normal;
				return true;
			}
		}
	}
	else if (LNS >= circle.m_radius)	// same code as above but opposite direction
	{
		CSD1130::Vector2D const p0 = lineSeg.m_pt0 + circle.m_radius * lineSeg.m_normal, p1 = lineSeg.m_pt1 + circle.m_radius * lineSeg.m_normal;
		CSD1130::Vector2D const vel = ptEnd - circle.m_center;
		CSD1130::Vector2D M{ vel.y, -vel.x }; CSD1130::Vector2DNormalize(M, M);
		if (CSD1130::Vector2DDotProduct(M, p0 - circle.m_center) * CSD1130::Vector2DDotProduct(M, p1 - circle.m_center) < 0.0f)
		{
			interTime = (CSD1130::Vector2DDotProduct(lineSeg.m_normal, lineSeg.m_pt0 - circle.m_center) + circle.m_radius) / CSD1130::Vector2DDotProduct(lineSeg.m_normal, vel);
			if (0.0f <= interTime && 1.0f >= interTime)
			{
				interPt = circle.m_center + vel * interTime;
				normalAtCollision = lineSeg.m_normal;
				return true;
			}
		}
	}
	return false; // no intersection
}

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
void CollisionResponse_CircleLineSegment(const CSD1130::Vec2 &ptInter, 
										 const CSD1130::Vec2 &normal,
										 CSD1130::Vec2 &ptEnd, 
										 CSD1130::Vec2 &reflected)
{
	CSD1130::Vector2D pen = ptEnd - ptInter;
	ptEnd = ptInter + pen - (2 * CSD1130::Vector2DDotProduct(normal, pen)) * normal;
	CSD1130::Vector2DNormalize(reflected, ptEnd - ptInter);
}
