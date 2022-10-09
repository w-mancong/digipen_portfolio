/*!*****************************************************************************
\file tddaa.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Assignment 2
\date 16-09-2022
\brief
This file contains function definition to dynamically allocate a 3D array
*******************************************************************************/
#include "tddaa.h"

using u64 = unsigned long long;

/*!*****************************************************************************
    \brief Dynamically allocate a 3D array

    \param [in] F: Specifies number of frames
    \param [in] R: Specifies number of rows in each frame
    \param [in] C: Specifies number of column within each row

    \return Pointer to the address of the first element to the dynamic 3D array
*******************************************************************************/
int*** allocate(int F, int R, int C) 
{
  int ***arr { nullptr }, **dptr{ nullptr }, *ptr{ nullptr };
  u64 const len = sizeof(int **) * F + sizeof(int *) * F * R + sizeof(int) * F * R * C;
  arr = new int **[len];

  dptr = reinterpret_cast<int **>(arr + F);
  ptr = reinterpret_cast<int *>(arr + F + F * R);
  u64 const index { static_cast<u64>(R * C) };
  for (int i = 0; i < F; ++i)
  {
    *(arr + i) = (dptr + R * i);
    for (int j = 0; j < R; ++j)
      *(*(arr + i) + j) = ((ptr + index * i) + C * j);
  }

  return arr;
}

/*!*****************************************************************************
    \brief Deallocate memory allocated on the heap
*******************************************************************************/
void deallocate( int ***ptr ) 
{
  delete[] ptr;
}
