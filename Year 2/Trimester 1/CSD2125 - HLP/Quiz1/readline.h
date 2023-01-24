/*!*****************************************************************************
\file readline.h
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Quiz 1
\date 31-08-22
\brief
This file contains a function readline that reads and return a single line.
Similar to std::getline in c++
*******************************************************************************/
#include <stdio.h>  // fopen/fclose, printf

/*!*****************************************************************************
    \brief
    Calling readline on a file stream containing characters 12345\n678 will
    result in the function returning a C-string 12345\0.

    \param [in] fileName: Pointer to file to be read

    \return A C-string with no newline character in it
*******************************************************************************/
char *readline(FILE *fileName);