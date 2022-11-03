/*!*****************************************************************************
\file matrix.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Quiz 4
\date 28-09-2022
\brief
Simple matrix functionalities
*******************************************************************************/
#include "matrix.h"
#include <iostream>
#include <iomanip>

/*!*****************************************************************************
    \brief

    \param [in] m: matrix to be printed
    \param [in] num_rows: Specifies number of rows
    \param [in] num_columns: Specifies number of column
*******************************************************************************/
void matrix_print(Matrix m, int num_rows, int num_columns)
{
  for (int i{}; i < num_rows; ++i)
  {
    for (int j{}; j < num_columns; ++j)
    {
      std::cout << std::setw(4) << m[i][j] << " ";
    }
    std::cout << "\n";
  }
}

/*!*****************************************************************************
    \brief Creates a dynamic 2D array

    \param [in] num_rows: Specifies number of rows
    \param [in] num_columns: Specifies number of columns

    \return Pointer to the address of the first element to the dynamic 2D array
*******************************************************************************/
Matrix matrix_create(int num_rows, int num_columns)
{
  Matrix ptr{nullptr};
  ptr = new int *[num_rows] {};
  for (int i{}; i < num_rows; ++i)
    *(ptr + i) = new int[num_columns]{};
  return ptr;
}

/*!*****************************************************************************
    \brief Adding two matrixes together

    \param [in] m1: First matrix to be added
    \param [in] m2: Second matrix to be added
    \param [out] result: Storing the result of the addition of the two matrix
*******************************************************************************/
void matrix_add(Matrix m1, Matrix m2, Matrix result, int num_rows, int num_columns)
{
  for (int i{}; i < num_rows; ++i)
  {
    for (int j{}; j < num_columns; ++j)
      *(*(result + i) + j) = *(*(m1 + i) + j) + *(*(m2 + i) + j);
  }
}

/*!*****************************************************************************
    \brief Return the transpose matrix of m

    \param [in] m: To construct the transpose of the matrix from
    \param [in] num_rows: Specifies the number of rows
    \param [in] num_columns: Specifies the number of columns

    \return Transpose matrix of m
*******************************************************************************/
Matrix matrix_transpose(Matrix m, int num_rows, int num_columns)
{
  Matrix res = matrix_create(num_columns, num_rows);

  for (int i{}; i < num_rows; ++i)
  {
    for (int j{}; j < num_columns; ++j)
      res[j][i] = m[i][j];
  }

  return res;
}

/*!*****************************************************************************
    \brief Deallocate memory of 2D dynamic array from the heap

    \param [in] m: To deallocate the memory from heap
    \param [in] num_rows: Specifies the number of rows
*******************************************************************************/
void matrix_delete(Matrix m, int num_rows)
{
  for (int i{}; i < num_rows; ++i)
    delete[] *(m + i);
  delete[] m;
}

/*!*****************************************************************************
    \brief Delete a row from the matrix

    \param [in] m: Matrix to have it's row to be deleted
    \param [in] r: Row of matrix to be deleted
    \param [in] num_columns: Specifies the number of columns
*******************************************************************************/
void matrix_delete_row( Matrix m, int r, int num_rows )
{
  delete[] *(m + r);
  for (int i{ r }; i < num_rows - 1; ++i)
    *(m + i) = *(m + i + 1);
}

/*!*****************************************************************************
    \brief Delete a row from the matrix

    \param [in] m: Matrix to have it's columns to be deleted
    \param [in] c: Column of matrix to be deleted
    \param [in] num_rows: Specifies the number of columns
    \param [in] num_columns: Specifies the number of columns
*******************************************************************************/
void matrix_delete_column( Matrix m, int c, int num_rows, int num_columns )
{
  for(int i{}; i < num_rows; ++i)
  {
    for (int j{c}; j < num_columns - 1; ++j)
      *(*(m + i) + j) = *(*(m + i) + j + 1);
  }
}