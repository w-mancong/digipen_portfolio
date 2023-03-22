/*!*****************************************************************************
\file		Vector2D.cpp
\author 	Wong Man Cong
\par    	DP email: w.mancong\@digipen.edu
\date   	26-03-2022
\brief
This file contains definitions for mathematic operations for Vectors in 2D

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*******************************************************************************/

#include <cmath>
#include "Vector2D.h"

namespace CSD1130
{
	/*!*****************************************************************************
	\brief
		Default constructor for Vector2D
	*******************************************************************************/
	Vector2D::Vector2D(void) : x{ 0.0f }, y{ 0.0f }{}

	/*!*****************************************************************************
	\brief
		Constructor for Vector2D that takes in x and y coordinate
	\param [in] _x:
		X coordinate
	\param [in] _y:
		Y coordinate
	*******************************************************************************/
	Vector2D::Vector2D(float _x, float _y) : x{ _x }, y{ _y } {}

	/*!*****************************************************************************
	\brief
		Overloaded += operator that adds the value of the two vector into this class
	\param [in] rhs:
		Vector to be added with this class
	\return
		A reference to this class
	*******************************************************************************/
	Vector2D& Vector2D::operator+=(const Vector2D& rhs)
	{
		x += rhs.x;
		y += rhs.y;
		return *this;
	}

	/*!*****************************************************************************
	\brief
		Overloaded -= operator that subtracts the value of the two vector into 
		this class
	\param [in] rhs:
		Vector to be subtracted with this class
	\return
		A reference to this class
	*******************************************************************************/
	Vector2D& Vector2D::operator-=(const Vector2D& rhs)
	{
		x -= rhs.x;
		y -= rhs.y;
		return *this;
	}

	/*!*****************************************************************************
	\brief
		Overloaded *= that multiplies this vector class with a scalar
	\param [in] rhs:
		Scalar to be multiplied with
	\return
		A reference to this class
	*******************************************************************************/
	Vector2D& Vector2D::operator*=(float rhs)
	{
		x *= rhs;
		y *= rhs;
		return *this;
	}

	/*!*****************************************************************************
	\brief
		Overloaded /= that divides this vector class with a scalar
	\param [in] rhs:
		Scalar to be divided with
	\return
		A reference to this class
	*******************************************************************************/
	Vector2D& Vector2D::operator/=(float rhs)
	{
		x /= rhs;
		y /= rhs;
		return *this;
	}

	/*!*****************************************************************************
	\brief
		Unary operator that negates the value of x and y coordinate
	\return 
		A copy of Vector2D after negating x and y coordinate
	*******************************************************************************/
	Vector2D Vector2D::operator-() const
	{
		return Vector2D(-x, -y);
	}

	/*!*****************************************************************************
	\brief
		Adds two vector lhs and rhs together
	\param [in] lhs:
		First vector to be added
	\param [in] rhs:
		Second vector to be added
	\return
		A copy of a Vector2D after adding lhs and rhs
	*******************************************************************************/
	Vector2D operator+(const Vector2D& lhs, const Vector2D& rhs)
	{
		return Vector2D(lhs.x + rhs.x, lhs.y + rhs.y);
	}

	/*!*****************************************************************************
	\brief
		Subtract vector lhs from rhs
	\param [in] lhs:
		First vector to be subtracted
	\param [in] rhs:
		Second vector to be subtracted
	\return 
		A copy of a Vector2D after subtracting lhs from rhs
	*******************************************************************************/
	Vector2D operator-(const Vector2D& lhs, const Vector2D& rhs)
	{
		return Vector2D(lhs.x - rhs.x, lhs.y - rhs.y);
	}

	/*!*****************************************************************************
	\brief
		Scalar multiplication of a Vector2D
	\param [in] lhs:
		Vector to be multiplied with
	\param [in] rhs:
		Scalar to multiply
	\return
		A copy of a Vector2D after scalar multiplication
	*******************************************************************************/
	Vector2D operator*(const Vector2D& lhs, float rhs)
	{
		return Vector2D(lhs.x * rhs, lhs.y * rhs);
	}

	/*!*****************************************************************************
	\brief
		Scalar multiplication of a Vector2D
	\param [in] lhs:
		Scalar to multiply
	\param [in] rhs:
		Vector to be multiplied with
	\return
		A copy of a Vector2D after scalar multiplication
	*******************************************************************************/
	Vector2D operator*(float lhs, const Vector2D& rhs)
	{
		return Vector2D(lhs * rhs.x, lhs * rhs.y);
	}

	/*!*****************************************************************************
	\brief
		Scalar division of a Vector2D
	\param [in] lhs:
		Vector to be divided with
	\param [in] rhs:
		Scalar to divide with
	\return
		A copy of a Vector2D after scalar division
	*******************************************************************************/
	Vector2D operator/(const Vector2D& lhs, float rhs)
	{
		return Vector2D(lhs.x / rhs, lhs.y / rhs);
	}

	/*!*****************************************************************************
	\brief
		Getting the normalize vector of pVec0 and storing it into pResult
	\param [in,out] pResult:
		Result of the normalized vector
	\param [in] pVec0:
		Vector that has yet to be normalized
	*******************************************************************************/
	void Vector2DNormalize(Vector2D& pResult, const Vector2D& pVec0)
	{
		float len = Vector2DLength(pVec0);
		pResult.x = pVec0.x / len, pResult.y = pVec0.y / len;
	}

	/*!*****************************************************************************
	\brief
		Getting the length of the vector pVec0
	\param [in] pVec0:
		Vector to find the length of
	\return
		Length of vector pVec0
	*******************************************************************************/
	float Vector2DLength(const Vector2D& pVec0)
	{
		return sqrtf(Vector2DSquareLength(pVec0));
	}

	/*!*****************************************************************************
	\brief
		Getting the square length of the vector pVec0
	\param [in] pVec0:
		Vector to find the square length of
	\return
		Square length of vector pVec0
	*******************************************************************************/
	float Vector2DSquareLength(const Vector2D& pVec0)
	{
		return pVec0.x * pVec0.x + pVec0.y * pVec0.y;
	}

	/*!*****************************************************************************
	\brief
		Takes in two vector and treat them as points and finding the distance
		between them
	\param [in] pVec0:
		First point
	\param [in] pVec1:
		Second point
	\return
		Distance between pVec0 and pVec1
	*******************************************************************************/
	float Vector2DDistance(const Vector2D& pVec0, const Vector2D& pVec1)
	{
		return sqrtf(Vector2DSquareDistance(pVec0, pVec1));
	}

	/*!*****************************************************************************
	\brief
		Takes in two vector and treat them as points and finding the square
		distance between them
	\param [in] pVec0:
		First point
	\param [in] pVec1:
		Second point
	\return
		Square distance between pVec0 and pVec1
	*******************************************************************************/
	float Vector2DSquareDistance(const Vector2D& pVec0, const Vector2D& pVec1)
	{
		Vector2D res{ pVec0.x - pVec1.x, pVec0.y - pVec1.y };
		return Vector2DSquareLength(res);
	}

	/*!*****************************************************************************
	\brief
		Dot product between two vectors pVec0 and pVec1
	\param [in] pVec0:
		First vector
	\param [in] pVec1:
		Second vector
	\return
		Scalar result of the dot product between pVec0 and pVec1
	*******************************************************************************/
	float Vector2DDotProduct(const Vector2D& pVec0, const Vector2D& pVec1)
	{
		return pVec0.x * pVec1.x + pVec0.y * pVec1.y;
	}

	/*!*****************************************************************************
	\brief
		Finding ad - bc of pVec0 and pVec1
	\param [in] pVec0: 
		Vector on the lhs
	\param [in] pVec1:
		Vector on the rhs
	\return
		ad - bc of pVec0 and pVec1
	*******************************************************************************/
	float Vector2DCrossProductMag(const Vector2D& pVec0, const Vector2D& pVec1)
	{
		return pVec0.x * pVec1.y - pVec0.y * pVec1.x;
	}
}