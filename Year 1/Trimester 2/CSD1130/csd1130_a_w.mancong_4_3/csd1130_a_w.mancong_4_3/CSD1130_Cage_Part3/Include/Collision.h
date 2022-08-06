/*!*****************************************************************************
\file		Collision.h
\author 	Wong Man Cong
\par    	DP email: w.mancong\@digipen.edu
\date   	04-04-2022
\brief
This file contains function declarations for  buidling a line segment, 
collision checking for line segment vs circle, dynamic vs static circle 
and dynamic vs dynamic circle

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
	CSD1130::Vec2	m_normal;												//Outward normalized normal
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
void BuildLineSegment(LineSegment& lineSegment,											//Line segment reference - input/output
	const CSD1130::Vec2& pos,											//Position - input
	float scale,														//Scale - input
	float dir);														//Direction - input

/*!*****************************************************************************
\brief
	Strut containing the relavent data members to store the position and radius
	of a circle
*******************************************************************************/
struct Circle
{
	CSD1130::Vec2	m_center;
	float			m_radius{ 1.0f };

	// Extra credits
	float			m_mass{ 1.0f };
};

/*!*****************************************************************************
\brief
	Struct containing the relavent data members to store starting position
	and direction of where a ray is going
*******************************************************************************/
struct Ray
{
	CSD1130::Vec2	m_pt0;
	CSD1130::Vec2	m_dir;
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
int CollisionIntersection_CircleLineSegment(const Circle& circle,			//Circle data - input
	const CSD1130::Vec2& ptEnd,												//End circle position - input
	const LineSegment& lineSeg,												//Line segment - input
	CSD1130::Vec2& interPt,													//Intersection point - output
	CSD1130::Vec2& normalAtCollision,										//Normal vector at collision time - output
	float& interTime,														//Intersection time ti - output
	bool& checkLineEdges);													//The last parameter is new - for Extra Credits: true = check collision with line segment edges

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
	const Circle &circle,													//Circle data - input
	const CSD1130::Vec2 &ptEnd,												//End circle position - input
	const LineSegment &lineSeg,												//Line segment - input
	CSD1130::Vec2 &interPt,													//Intersection point - output
	CSD1130::Vec2 &normalAtCollision,										//Normal vector at collision time - output
	float &interTime);														//Intersection time ti - output

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
	const CSD1130::Vec2 &velA,												//CircleA velocity - input
	const Circle &circleB,													//CircleB data - input
	const CSD1130::Vec2 &velB,												//CircleA velocity - input
	CSD1130::Vec2 &interPtA,												//Intersection point of CircleA at collision time - output
	CSD1130::Vec2 &interPtB,												//Intersection point of CircleB at collision time - output
	float &interTime);														//intersection time - output

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
int CollisionIntersection_RayCircle(const Ray &ray,							//A ray containing the data of the moving dot - input
	const Circle &circle,													//Static circle data - input
	float &interTime);														//Intersection time - output

// RESPONSE FUNCTIONS
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
	CSD1130::Vec2& reflected);															//Normalized reflection vector direction - output

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
void CollisionResponse_CirclePillar(const CSD1130::Vec2& normal,						//Normal vector of reflection on collision time - input
	const float &interTime,																//Intersection time - input
	const CSD1130::Vec2& ptStart,														//Starting position of the circle - input
	const CSD1130::Vec2& ptInter,														//Intersection position of the circle - input
	CSD1130::Vec2& ptEnd,																//Final position of the circle after reflection - output
	CSD1130::Vec2& reflectedVectorNormalized);											//Normalized reflection vector - output

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
void CollisionResponse_CircleCircle(CSD1130::Vec2 &normal,								//Normal vector of reflection on collision time - input
	const float interTime,																//Intersection time - input
	CSD1130::Vec2 &velA,																//Velocity of CircleA - input
	const float &massA,																	//Mass of CircleA - input
	CSD1130::Vec2 &interPtA,															//Intersection position of circle A at collision time - input
	CSD1130::Vec2 &velB,																//Velocity of CircleB - input
	const float &massB,																	//Mass of CircleB - input
	CSD1130::Vec2& interPtB,															//Intersection position of circle B at collision time - input
	CSD1130::Vec2 &reflectedVectorA,													//Non-Normalized reflected vector of Circle A - output
	CSD1130::Vec2 &ptEndA,																//Final position of the circle A after reflection - output
	CSD1130::Vec2 &reflectedVectorB,													//Non-Normalized reflected vector of Circle B - output
	CSD1130::Vec2 &ptEndB);																//Final position of the circle B after reflection - output



#endif // CSD1130_COLLISION_H_