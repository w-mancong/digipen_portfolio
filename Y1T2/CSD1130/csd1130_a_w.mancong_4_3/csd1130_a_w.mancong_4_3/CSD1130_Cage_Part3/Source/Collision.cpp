/*!*****************************************************************************
\file		Collision.cpp
\author 	Wong Man Cong
\par    	DP email: w.mancong\@digipen.edu
\date   	04-04-2022
\brief
This file contains function definition for building a line segment, 
collision checking for line segment vs circle, dynamic vs static circle
and dynamic vs dynamic circle

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*******************************************************************************/

#include <algorithm>
#define NOMINMAX
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
void BuildLineSegment(LineSegment& lineSegment,
	const CSD1130::Vec2& pos,
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
int CollisionIntersection_CircleLineSegment(const Circle& circle,			//Circle data - input
	const CSD1130::Vec2& ptEnd,												//End circle position - input
	const LineSegment& lineSeg,												//Line segment - input
	CSD1130::Vec2& interPt,													//Intersection point - output
	CSD1130::Vec2& normalAtCollision,										//Normal vector at collision time - output
	float& interTime,														//Intersection time ti - output
	bool& checkLineEdges)													//The last parameter is new - for Extra Credits: true = check collision with line segment edges
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
	if (!checkLineEdges)
		return false;
	return CheckMovingCircleToLineEdge(true, circle, ptEnd, lineSeg, interPt, normalAtCollision, interTime);
}

/*!*****************************************************************************
\brief
	Check for collision between a circle and line segment's edge
\param [in] withinBothLines:
	A flag to check if circle is starting between two imaginary line segment
\param [in] circle:
	Data containing the circle to check collision with
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
	True if there is a collision between the circle and line segment's edge, 
	else false
*******************************************************************************/
int CheckMovingCircleToLineEdge(bool withinBothLines,						//Flag stating that the circle is starting from between 2 imaginary line segments distant +/- Radius respectively - input
	const Circle& circle,													//Circle data - input
	const CSD1130::Vec2& ptEnd,												//End circle position - input
	const LineSegment& lineSeg,												//Line segment - input
	CSD1130::Vec2& interPt,													//Intersection point - output
	CSD1130::Vec2& normalAtCollision,										//Normal vector at collision time - output
	float& interTime)														//Intersection time ti - output
{
	if (!withinBothLines) 
		return false;
	CSD1130::Vector2D const vel = ptEnd - circle.m_center; CSD1130::Vector2D vel_norm; CSD1130::Vector2DNormalize(vel_norm, vel);
	CSD1130::Vector2D M{ vel.y, -vel.x }; CSD1130::Vector2DNormalize(M, M);
	if (CSD1130::Vector2DDotProduct(lineSeg.m_pt0 - circle.m_center, lineSeg.m_pt1 - lineSeg.m_pt0) > 0.0f) // p0 side
	{
		float const m = CSD1130::Vector2DDotProduct(lineSeg.m_pt0 - circle.m_center, vel_norm);
		if (0.0f < m)
		{
			float const dist0 = CSD1130::Vector2DDotProduct(lineSeg.m_pt0 - circle.m_center, M);
			if (std::abs(dist0) > circle.m_radius) // no collision
				return false;
			float const s = std::sqrtf(circle.m_radius * circle.m_radius - dist0 * dist0);
			interTime = (m - s) / CSD1130::Vector2DLength(vel);
			if (1.0f >= interTime)
			{
				interPt = circle.m_center + vel * interTime;
				CSD1130::Vector2DNormalize(normalAtCollision, circle.m_center - lineSeg.m_pt0);
				return true;
			}
		}
	}
	else if (CSD1130::Vector2DDotProduct(lineSeg.m_pt1 - circle.m_center, lineSeg.m_pt1 - lineSeg.m_pt0) < 0.0f) // p1 side
	{
		float const m = CSD1130::Vector2DDotProduct(lineSeg.m_pt1 - circle.m_center, vel_norm);
		if (0.0f < m)
		{
			float const dist1 = CSD1130::Vector2DDotProduct(lineSeg.m_pt1 - circle.m_center, M);
			if (std::abs(dist1) > circle.m_radius) // no collision
				return false;
			float const s = std::sqrtf(circle.m_radius * circle.m_radius - dist1 * dist1);
			interTime = (m - s) / CSD1130::Vector2DLength(vel);
			if (1.0f >= interTime)
			{
				interPt = circle.m_center + vel * interTime;
				CSD1130::Vector2DNormalize(normalAtCollision, circle.m_center - lineSeg.m_pt1);
				return true;
			}
		}
	}
	else
	{
		bool p0Side = false;
		float const dist0 = CSD1130::Vector2DDotProduct(lineSeg.m_pt0 - circle.m_center, M), dist1 = CSD1130::Vector2DDotProduct(lineSeg.m_pt1 - circle.m_center, M);
		float const abs_dist0 = std::abs(dist0), abs_dist1 = std::abs(dist1);
		if (abs_dist0 > circle.m_radius && abs_dist1 > circle.m_radius)
			return false;
		else if (abs_dist0 <= circle.m_radius && abs_dist1 <= circle.m_radius)
		{
			float const m0 = CSD1130::Vector2DDotProduct(lineSeg.m_pt0 - circle.m_center, vel_norm), m1 = CSD1130::Vector2DDotProduct(lineSeg.m_pt1 - circle.m_center, vel_norm);
			float const abs_m0 = std::abs(m0), abs_m1 = std::abs(m1);
			if (abs_m0 < abs_m1)
				p0Side = true;
		}
		else if (abs_dist0 <= circle.m_radius)
			p0Side = true;
		if (p0Side) // circle is closer to p0
		{
			float const m = CSD1130::Vector2DDotProduct(lineSeg.m_pt0 - circle.m_center, vel_norm);
			if (0.0f > m)	// no collision
				return false;
			float const s = std::sqrtf(circle.m_radius * circle.m_radius - dist0 * dist0);
			interTime = (m - s) / CSD1130::Vector2DLength(vel);
			if (1.0f >= interTime)
			{
				interPt = circle.m_center + vel * interTime;
				CSD1130::Vector2DNormalize(normalAtCollision, interPt - lineSeg.m_pt0);
				return true;
			}
		}
		else
		{
			float const m = CSD1130::Vector2DDotProduct(lineSeg.m_pt1 - circle.m_center, vel_norm);
			if (0.0f > m)	// no collision
				return false;
			float const s = std::sqrtf(circle.m_radius * circle.m_radius - dist1 * dist1);
			interTime = (m - s) / CSD1130::Vector2DLength(vel);
			if (1.0f >= interTime)
			{
				interPt = circle.m_center + vel * interTime;
				CSD1130::Vector2DNormalize(normalAtCollision, interPt - lineSeg.m_pt1);
				return true;
			}
		}
	}
	return false;
}

/*!*****************************************************************************
\brief
	Checking for intersection between two circles, be it static or dynamic
\param [in] circleA:
	First circle to check collision with
\param [in] velA:
	Velocity of first circle
\param [in] circleB:
	Second circle to check collision with
\param [in] velB:
	Velocity of second circle
\param [out] interPtA:
	Intersection point of circleA at collision time
\param [out] interPtB:
	Intersection point of circleB at collision time
\param [out] interTime:
	Time of intersection between the two circle
\return 
	True if there is an intersection between the two circle, else false
*******************************************************************************/
int CollisionIntersection_CircleCircle(const Circle &circleA,				//CircleA data - input
	const CSD1130::Vec2& velA,												//CircleA velocity - input
	const Circle& circleB,													//CircleB data - input
	const CSD1130::Vec2& velB,												//CircleB velocity - input
	CSD1130::Vec2& interPtA,												//Intersection point of CircleA at collision time - output
	CSD1130::Vec2& interPtB,												//Intersection point of CircleB at collision time - output
	float& interTime)														//intersection time - output
{
	CSD1130::Vector2D const rv = velA - velB;		// relative velocity (making B my static circle)
	Ray const ray{ circleA.m_center, rv }; Circle const circleC{ circleB.m_center, circleA.m_radius + circleB.m_radius };
	bool const COLLIDED = CollisionIntersection_RayCircle(ray, circleC, interTime);
	interPtA = circleA.m_center + velA * interTime;
	interPtB = circleB.m_center + velB * interTime;
	return COLLIDED;
}

/*!*****************************************************************************
\brief
	Checking for intersection between a ray and circle
\param [in] ray:
	Ray data containing the starting point and direction of where the ray is going
\param [in] circle:
	Data containing the circle to check collision with ray
\param [in] interTime
	Time of intersection between the circle and ray
\return
	True if there is an intersection between the ray and circle, else false
*******************************************************************************/
int CollisionIntersection_RayCircle(const Ray& ray,							//A ray containing the data of the moving dot - input
	const Circle& circle,													//Static circle data - input
	float& interTime)														//Intersection time - output
{
	CSD1130::Vector2D const rayToCircleCenter = circle.m_center - ray.m_pt0;
	CSD1130::Vector2D normal;  CSD1130::Vector2DNormalize(normal, ray.m_dir);
	float const m		= CSD1130::Vector2DDotProduct(rayToCircleCenter, normal);
	float const dist0	= CSD1130::Vector2DDotProduct(rayToCircleCenter, rayToCircleCenter) - (m * m);	// Pythagoras theorem 
	float const sq_dist = CSD1130::Vector2DSquareLength(rayToCircleCenter);								// To check if ray's position is inside of circle
	float const r2		= circle.m_radius * circle.m_radius;											// squared length of radius
	// Test rejection 1 and 2
	if ((0.0f > m && sq_dist > r2) || dist0 > r2)
		return false;
	float const s = sqrtf(r2 - dist0);
	interTime = std::min((m - s) / CSD1130::Vector2DLength(ray.m_dir), (m + s) / CSD1130::Vector2DLength(ray.m_dir));
	if (0.0f <= interTime && 1.0f >= interTime)
		return true;
	return false;
}


/*!*****************************************************************************
\brief
	Calculate the new end position and the reflected vector of the circle
	after colliding with the line segment
\param [in] ptInter:
	The point of intersection between the circle and line segment
\param [in] normal:
	Normal vector at the point of intersection between the circle and line segment
\param [in,out] ptEnd:
	Used to calculate the penetration and to store the new position after
\param [out] reflected:
	Vector2D used to store the reflected vector
*******************************************************************************/
void CollisionResponse_CircleLineSegment(const CSD1130::Vec2& ptInter,					//Intersection position of the circle - input
	const CSD1130::Vec2& normal,														//Normal vector of reflection on collision time - input
	CSD1130::Vec2& ptEnd,																//Final position of the circle after reflection - output
	CSD1130::Vec2& reflected)															//Normalized reflection vector direction - output
{
	CSD1130::Vector2D pen = ptEnd - ptInter;
	ptEnd = ptInter + pen - (2 * CSD1130::Vector2DDotProduct(normal, pen)) * normal;
	CSD1130::Vector2DNormalize(reflected, ptEnd - ptInter);
}

/*!*****************************************************************************
\brief
	Calculate the new end position and the reflected vector of the circle
	after colliding with a static circle
\param [in] normal:
	Normal vector of reflection on collision time between the two circle
\param [in] interTime:
	Time of intersection between the static and dynamic circle
\param [in] ptStart:
	Starting position of the dynamic circle
\param [in] ptInter:
	Intersection point of the dynamic circle
\param [out] ptEnd:
	Vector use to store the new position of dynamic circle after reflection
\param [out] reflectedVectorNormalized:
	Vector use to store the normalized reflected vector 
*******************************************************************************/
void CollisionResponse_CirclePillar(const CSD1130::Vec2& normal,					//Normal vector of reflection on collision time - input
	const float& interTime,															//Intersection time - input
	const CSD1130::Vec2& ptStart,													//Starting position of the circle - input
	const CSD1130::Vec2& ptInter,													//Intersection position of the circle - input
	CSD1130::Vec2& ptEnd,															//Final position of the circle after reflection - output
	CSD1130::Vec2& reflectedVectorNormalized)										//Normalized reflection vector - output
{
	CSD1130::Vector2D const m = ptStart - ptInter;
	reflectedVectorNormalized = 2.0f * (CSD1130::Vector2DDotProduct(m, normal)) * normal - m;
	CSD1130::Vector2DNormalize(reflectedVectorNormalized, reflectedVectorNormalized);
	float const k = CSD1130::Vector2DLength((ptInter - ptStart) / interTime);
	ptEnd = ptInter + k * reflectedVectorNormalized * (1.0f - interTime);
}

/*!*****************************************************************************
\brief
	Calculate the new end position and reflected vector for both the dynamic
	circle after collision with respect to mass
\param [in] normal:
	Normal vector of reflection on collision time
\param [in] interTime:
	Time of intersection between the two dynamic circles
\param [in] velA:
	Velocity of first circle
\param [in] massA:
	Mass of first circle
\param [in] interPtA:
	Intersection point of first circle
\param [in] velB:
	Velocity of second circle
\param [in] massB:
	Mass of second circle
\param [in] interPtB:
	Intersection point of second circle
\param [out] reflectedVectorA:
	Non-normalized reflected vector of first circle after reflection
\param [out] ptEndA:
	New position of the first circle after reflection
\param [out] reflectedVectorB:
	Non-normalized reflect vector of the second circle after reflection
\param [out] ptEndB:
	New position of the second circle after reflection
*******************************************************************************/
void CollisionResponse_CircleCircle(CSD1130::Vec2& normal,								//Normal vector of reflection on collision time - input
	const float interTime,																//Intersection time - input
	CSD1130::Vec2& velA,																//Velocity of CircleA - input
	const float& massA,																	//Mass of CircleA - input
	CSD1130::Vec2& interPtA,															//Intersection position of circle A at collision time - input
	CSD1130::Vec2& velB,																//Velocity of CircleB - input
	const float& massB,																	//Mass of CircleB - input
	CSD1130::Vec2& interPtB,															//Intersection position of circle B at collision time - input
	CSD1130::Vec2& reflectedVectorA,													//Non-Normalized reflected vector of Circle A - output
	CSD1130::Vec2& ptEndA,																//Final position of the circle A after reflection - output
	CSD1130::Vec2& reflectedVectorB,													//Non-Normalized reflected vector of Circle B - output
	CSD1130::Vec2& ptEndB)																//Final position of the circle B after reflection - output
{
	float const aA = CSD1130::Vector2DDotProduct(velA, normal), aB = CSD1130::Vector2DDotProduct(velB, normal);
	reflectedVectorA = velA - (2.0f * (aA - aB) / (massA + massB)) * massB * normal;
	reflectedVectorB = velB + (2.0f * (aA - aB) / (massA + massB)) * massA * normal;
	ptEndA = interPtA + (1.0f - interTime) * reflectedVectorA;
	ptEndB = interPtB + (1.0f - interTime) * reflectedVectorB;
}
