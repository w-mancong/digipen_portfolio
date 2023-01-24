/*!*****************************************************************************
\file readline.c
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
#include "readline.h"
#include <string.h>
#include <stdlib.h> // malloc/free

/*!*****************************************************************************
    \brief
    Calling readline on a file stream containing characters 12345\n678 will
    result in the function returning a C-string 12345\0.

    \param [in] fileName: Pointer to file to be read
    
    \return A C-string with no newline character in it
*******************************************************************************/
char *readline(FILE *fileName)
{
    int capacity = 2;
    char *str = (char *)malloc(capacity * sizeof(char));
    size_t len = 0;

    // loop through the entire file, if eof is encountered, break the loop
    while (fgets(str + len, capacity - len, fileName) != NULL)
    {
        len = strlen(str); capacity <<= 1;
        if (*(str + len - 1) == '\n')
        {
            *(str + len - 1) = '\0';
            break;
        }
        str = (char *)realloc(str, capacity);
    }

    return str;
}