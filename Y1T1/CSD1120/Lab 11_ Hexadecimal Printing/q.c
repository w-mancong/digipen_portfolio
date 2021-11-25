/*!
@file       q.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@tutorial   11
@date       25/11/2021
@brief      printing of hexa values
*//*__________________________________________________________________________________________________________________________*/
#include <stdio.h>
#include "q.h"

/*!
@brief  Display binary data in hexadecimal form and their ASCII counterparts side by side

@param  ptr: Address to the first byte
        size: Total count of bytes
        span: Count of bytes to be printed out per line
*//*__________________________________________________________________________________________________________________________*/
void print_data(const void* ptr, size_t size, size_t span)
{
	for (size_t i = 0; i < size / 4; i += span / 4)
	{
		for (size_t j = 0; j < span; ++j)
			printf("%x%s%s", (*((char*)ptr + j) + (char)i), !((j + 1) % 4) ? "   " : " ", (j + 1) == span ? "|   " : "");
		for (size_t j = 0; j < span; ++j)
			printf("%c%s%s", (*((char*)ptr + j) + (char)i), !((j + 1) % 4) && (j + 1) != span ? "   " : (j + 1) == span ? "" : " ", (j + 1) == span ? "\n" : "");
	}
}