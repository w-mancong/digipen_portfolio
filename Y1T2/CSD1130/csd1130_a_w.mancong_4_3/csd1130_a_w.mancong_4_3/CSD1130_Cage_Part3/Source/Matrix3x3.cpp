/*!*****************************************************************************
\file		Matrix3x3.cpp
\author 	Wong Man Cong
\par    	DP email: w.mancong\@digipen.edu
\date   	04-04-2022
\brief
This file contains definitions for mathematic operations for a 3x3 Matrix

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*******************************************************************************/

#define _USE_MATH_DEFINES
#include <cmath>
#include <memory>
#include "Matrix3x3.h"

namespace
{
	const size_t SIZE = 9; // 3x3 matrix

	/*!*****************************************************************************
	\brief
		Helper function to get grid based on row, col and max column of the array
	\param [in] ptr:
		Address of the first element of the array
	\param [in] row:
		Row index
	\param [in] col:
		Column index
	\param [in] WIDTH:
		Maximum column of the array
	\return
		Const reference to a float based on row, column index and the maximum column
		of the array
	*******************************************************************************/
	float const& GetGrid(float const* ptr, size_t row, size_t col, size_t WIDTH = 3)
	{
		return *(ptr + row * WIDTH + col);
	}

	/*!*****************************************************************************
	\brief
		Helper function to get grid based on row, col and max column of the array
	\param [in] ptr:
		Address of the first element of the array
	\param [in] row:
		Row index
	\param [in] col:
		Column index
	\param [in] WIDTH:
		Maximum column of the array
	\return
		Reference to a float based on row, column index and the maximum column
		of the array
	*******************************************************************************/
	float& GetGrid(float* ptr, size_t row, size_t col, size_t WIDTH = 3)
	{
		return const_cast<float&>(GetGrid(static_cast<float const*>(ptr), row, col, WIDTH));
	}

	/*!*****************************************************************************
	\brief
		Helper function to determine the new matrix to find it's determinant
	\param [in,out] dst:
		Destination to store values of new matrix
	\param [in] src:
		Source of matrix that dst is based of
	\param [in] row:
		Row to be ignored
	\param [in] col:
		Column to be ignored
	*******************************************************************************/
	void BarMatrix(float* dst, float const* src, size_t row, size_t col)
	{
		for (size_t i = 0, r = 0; r < 2; ++i)
		{
			if (i == row)
				continue;
			for (size_t j = 0, c = 0; c < 2; ++j)
			{
				if (j == col)
					continue;
				GetGrid(dst, r, c++, 2) = GetGrid(src, i, j, 3);
			}
			++r;
		}
	}

	/*!*****************************************************************************
	\brief
		Helper function to find the determinant of ptr
	\param [in] ptr:
		Address of the first element of the array (3x3 or 2x2 matrix)
	\param [in] n:
		N x N of the matrix
	\return
		Final determinant of ptr
	*******************************************************************************/
	float Determinant(float const* ptr, size_t n)
	{
		if (n == 1)
			return GetGrid(ptr, 0, 0, 1);
		if (n == 2)	// ad - bc
			return GetGrid(ptr, 0, 0, 2) * GetGrid(ptr, 1, 1, 2) - GetGrid(ptr, 0, 1, 2) * GetGrid(ptr, 1, 0, 2);
		float det = {};
		float tmp[4] = {};
		for (size_t j = 0; j < 3; ++j)
		{
			BarMatrix(tmp, ptr, 0, j);
			det += powf(-1.0, static_cast<float>(j)) * GetGrid(ptr, 0, j, 3) * Determinant(tmp, 2);
		}
		return det;
	}
}

namespace CSD1130
{
	/*!*****************************************************************************
	\brief
		Default constructor for Matrix3x3
	*******************************************************************************/
	Matrix3x3::Matrix3x3(void)
	{
		memset(m, 0, sizeof(m));
	}

	/*!*****************************************************************************
	\brief
		Constructor of Matrix3x3 that converts the float const* to a Matrix3x3
	\param [in] pArr:
		Address of the first element to the array
	*******************************************************************************/
	Matrix3x3::Matrix3x3(const float* pArr)
	{
		for (size_t i = 0; i < SIZE; ++i)
			*(m + i) = *(pArr + i);
	}

	/*!*****************************************************************************
	\brief
		Constructor that takes in 9 floats of individual values for the 3x3 Matrix
	\param [in] _00:
		m00
	\param [in] _01:
		m01
	\param [in] _02:
		m02
	\param [in] _10:
		m10
	\param [in] _11:
		m11
	\param [in] _12:
		m12
	\param [in] _20:
		m20
	\param [in] _21:
		m21
	\param [in] _22:
		m22
	*******************************************************************************/
	Matrix3x3::Matrix3x3(	float _00, float _01, float _02,
							float _10, float _11, float _12,
							float _20, float _21, float _22) 
						:	m00{ _00 }, m01{ _01 }, m02{ _02 },
							m10{ _10 }, m11{ _11 }, m12{ _12 }, 
							m20{ _20 }, m21{ _21 }, m22{ _22 } {}

	/*!*****************************************************************************
	\brief
		Copy constructor for Matrix3x3
	\param [in] rhs:
		Matrix to copy it's data from
	\return
		A reference to this class
	*******************************************************************************/
	Matrix3x3& Matrix3x3::operator=(const Matrix3x3& rhs)
	{
		for (size_t i = 0; i < SIZE; ++i)
			*(m + i) = *(rhs.m + i);
		return *this;
	}

	/*!*****************************************************************************
	\brief
		Overloaded operator *= that performs matrix multiplication with rhs
	\param [in] rhs:
		Matrix to be multiplied with
	\return
		A reference to this class
	*******************************************************************************/
	Matrix3x3& Matrix3x3::operator*=(const Matrix3x3& rhs)
	{
		return (*this = *this * rhs);
	}

	/*!*****************************************************************************
	\brief
		Matrix multiplication of lhs and rhs
	\param [in] lhs:
		First matrix to be multiplied with
	\param [in] rhs:
		Second matrix to be multiplied with
	\return
		A copy of a matrix after matrix lhs * rhs
	*******************************************************************************/
	Matrix3x3 operator*(const Matrix3x3& lhs, const Matrix3x3& rhs)
	{
		const size_t rows = 3, cols = 3, final_size = rows * cols;
		size_t curr_size = 0, k = 0, l = 0;
		float tmp[final_size] = { 0.0f };

		while (curr_size < final_size)
		{
			for (size_t i = 0; i < rows; ++i)
			{
				float sum = {};
				for (size_t j = 0; j < cols; ++j)
					sum += GetGrid(lhs.m, l, j) * GetGrid(rhs.m, j, k);
				GetGrid(tmp, l, k++) = sum, ++curr_size;
			}
			// reset condition when im done with the current col
			if (!(curr_size % cols)) 
				++l, k = 0;
		}
		return Matrix3x3{ tmp };
	}

	/*!*****************************************************************************
	\brief
		Matrix multiplication with a Vector2D
	\param [in] pMtx:
		Matrix to be multiplied with
	\param [in] rhs:
		Vector to be multiplied with
	\return
		A copy of a vector after matrix multiplication with a vector
	*******************************************************************************/
	Vector2D operator*(const Matrix3x3& pMtx, const Vector2D& rhs)
	{
		float vec[3] = { rhs.x, rhs.y, 1.0f };
		const size_t cols = 3, final_size = cols;
		size_t curr_size = 0, k = 0, l = 0;
		float tmp[final_size] = { 0.0f };

		while (curr_size < final_size)
		{
			float sum = {};
			for (size_t j = 0; j < cols; ++j)
				sum += GetGrid(pMtx.m, l, j) * vec[j];
			tmp[k++] = sum, ++l, ++curr_size;
		}
		return Vector2D{ tmp[0], tmp[1] };
	}

	/*!*****************************************************************************
	\brief
		Matrix scalar multiplication
	\param [in] lhs:
		Matrix to be multiplied
	\param [in] rhs:
		Scalar to be multiplied with
	\return
		A copy of a matrix after matrix scalar multiplication
	*******************************************************************************/
	Matrix3x3 operator*(Matrix3x3 const& lhs, float rhs)
	{
		Matrix3x3 tmp{ lhs };
		for (size_t i = 0; i < 3; ++i)
		{
			for (size_t j = 0; j < 3; ++j)
				GetGrid(tmp.m, i, j, 3) *= rhs;
		}
		return tmp;
	}

	/*!*****************************************************************************
	\brief
		Turns pResults into an indentity matrix
	\param [in,out] pResult:
		Storing the identity matrix into pResult
	*******************************************************************************/
	void Mtx33Identity(Matrix3x3& pResult)
	{
		memset(pResult.m, 0, sizeof(pResult.m));
		for (size_t i = 0; i < 3; ++i)
			GetGrid(pResult.m, i, i) = 1.0f;
	}

	/*!*****************************************************************************
	\brief
		Turn pResult into a translation matrix based on x and y
	\param [in,out] pResult:
		Storing the translation matrix into pResult
	\param [in] x:
		X coordinate
	\param [in] y:
		Y coordinate
	*******************************************************************************/
	void Mtx33Translate(Matrix3x3& pResult, float x, float y)
	{
		Mtx33Identity(pResult);
		pResult.m02 = x, pResult.m12 = y;
	}

	/*!*****************************************************************************
	\brief
		Turn pResult into a scale matrix based on x and y
	\param [in,out] pResult:
		Storing the scale matrix into pResult
	\param [in] x:
		X scale
	\param [in] y:
		Y scale
	*******************************************************************************/
	void Mtx33Scale(Matrix3x3& pResult, float x, float y)
	{
		Mtx33Identity(pResult);
		pResult.m00 = x, pResult.m11 = y;
	}

	/*!*****************************************************************************
	\brief
		Turn pResult into a rotation matrix based on angle (in Radians)
	\param [in,out] pResult:
		Storing the rotation matrix into pResult
	\param [in] angle:
		Angle in radians
	*******************************************************************************/
	void Mtx33RotRad(Matrix3x3& pResult, float angle)
	{
		Mtx33Identity(pResult);
		const float cos = cosf(angle), sin = sinf(angle);
		pResult.m00 = cos, pResult.m01 = -sin;
		pResult.m10 = sin, pResult.m11 = cos;
	}

	/*!*****************************************************************************
	\brief
		Turn pResult into a rotation matrix based on angle (in Degrees)
	\param [in,out] pResult:
		Storing the rotation matrix into pResult
	\param [in] angle:
		Angle in degrees
	*******************************************************************************/
	void Mtx33RotDeg(Matrix3x3& pResult, float angle)
	{
		const float RAD = angle * static_cast<float>(M_PI) / 180.0f;
		Mtx33RotRad(pResult, RAD);
	}

	/*!*****************************************************************************
	\brief
		Storing the transpose matrix of pMtx into pResult
		Transpose: Swapping the N row of the matrix to the N column of the matrix
	\param [in,out] pResult:
		Storing the transpose matrix into pResult
	\param [in] pMtx:
		Getting the tranpose matrix of pMtx
	*******************************************************************************/
	void Mtx33Transpose(Matrix3x3& pResult, const Matrix3x3& pMtx)
	{
		float tmp[SIZE] = {};
		const size_t rows = 3, cols = 3;
		for (size_t i = 0; i < rows; ++i)
		{
			for (size_t j = 0; j < cols; ++j)
				GetGrid(tmp, j, i) = GetGrid(pMtx.m, i, j);
		}
		memcpy_s(pResult.m, sizeof(pResult.m), tmp, sizeof(tmp));
	}

	/*!*****************************************************************************
	\brief
		Finding the inverse matrix of pMtx and storing it into pResult
	\param [in,out] pResult:
		Storing the inverse matrix into pResult
	\param [in,out] determinant:
		Storing the determinant of pMtx and using it at the end to find the inverse
		matrix
	\param [in] pMtx:
		Finding the inverse matrix of pMtx
	*******************************************************************************/
	void Mtx33Inverse(Matrix3x3* pResult, float* determinant, const Matrix3x3& pMtx)
	{
		*determinant = Determinant(pMtx.m, 3);
		if (!*determinant)
			return;
		float inv[9] = {}, tmp[4] = {};
		float flag = 1.0f;
		for (size_t i = 0; i < 3; ++i)
		{
			for (size_t j = 0; j < 3; ++j)
			{
				BarMatrix(tmp, pMtx.m, i, j);
				GetGrid(inv, i, j, 3) = Determinant(tmp, 2) * flag;
				flag *= -1.0f;
			}
		}
		Matrix3x3 inverse{ inv };
		Mtx33Transpose(inverse, inverse);
		inverse = inverse * (1.0f / *determinant);
		memcpy_s(pResult->m, sizeof(pResult->m), inverse.m, sizeof(inverse.m));
	}
}