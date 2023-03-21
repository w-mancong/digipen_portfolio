/*!*****************************************************************************
\file		Vector2D.h
\author 	Wong Man Cong
\par    	DP email: w.mancong\@digipen.edu
\date   	26-03-2022
\brief
This file contains definitions for mathematic operations for Vectors in 2D

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*******************************************************************************/

#ifndef CSD1130_VECTOR2D_H_
#define CSD1130_VECTOR2D_H_

namespace CSD1130
{
	#ifdef _MSC_VER
	// Supress warning: nonstandard extension used : nameless struct/union
	#pragma warning( disable : 4201 )
	#endif

	/**************************************************************************/
	/*!

	 */
	/**************************************************************************/
	typedef union Vector2D
	{
		struct
		{
			float x, y;
		};

		float m[2];

		// Constructors
		Vector2D();
		Vector2D(float _x, float _y);

		// Assignment operators
		Vector2D& operator += (const Vector2D &rhs);
		Vector2D& operator -= (const Vector2D &rhs);
		Vector2D& operator *= (float rhs);
		Vector2D& operator /= (float rhs);

		// Unary operators
		Vector2D operator -() const;

	} Vector2D, Vec2, Point2D, Pt2;


	#ifdef _MSC_VER
	// Supress warning: nonstandard extension used : nameless struct/union
	#pragma warning( default : 4201 )
	#endif

	// Binary operators
	Vector2D operator + (const Vector2D &lhs, const Vector2D &rhs);
	Vector2D operator - (const Vector2D &lhs, const Vector2D &rhs);
	Vector2D operator * (const Vector2D &lhs, float rhs);
	Vector2D operator * (float lhs, const Vector2D &rhs);
	Vector2D operator / (const Vector2D &lhs, float rhs);

	/**************************************************************************/
	/*!
		In this function, pResult will be the unit vector of pVec0
	 */
	/**************************************************************************/
	void	Vector2DNormalize(Vector2D &pResult, const Vector2D &pVec0);
	
	/**************************************************************************/
	/*!
		This function returns the length of the vector pVec0 
	 */
	/**************************************************************************/
	float	Vector2DLength(const Vector2D &pVec0);
	
	/**************************************************************************/
	/*!
		This function returns the square of pVec0's length. Avoid the square root 
	 */
	/**************************************************************************/
	float	Vector2DSquareLength(const Vector2D &pVec0);
	
	/**************************************************************************/
	/*!
		In this function, pVec0 and pVec1 are considered as 2D points.
		The distance between these 2 2D points is returned
	 */
	/**************************************************************************/
	float	Vector2DDistance(const Vector2D &pVec0, const Vector2D &pVec1);
	
	/**************************************************************************/
	/*!
		In this function, pVec0 and pVec1 are considered as 2D points.
		The squared distance between these 2 2D points is returned. 
		Avoid the square root
	 */
	/**************************************************************************/
	float	Vector2DSquareDistance(const Vector2D &pVec0, const Vector2D &pVec1);
	
	/**************************************************************************/
	/*!
		This function returns the dot product between pVec0 and pVec1
	 */
	/**************************************************************************/
	float	Vector2DDotProduct(const Vector2D &pVec0, const Vector2D &pVec1);
	
	/**************************************************************************/
	/*!
		This function returns the cross product magnitude 
		between pVec0 and pVec1
	 */
	/**************************************************************************/
	float	Vector2DCrossProductMag(const Vector2D &pVec0, const Vector2D &pVec1);
}

#endif // CSD1130_VECTOR2D_H_