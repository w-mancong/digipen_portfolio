/*!
@file       add_digit.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@tutorial   12
@date       02/12/2021
@brief      big number addition calculator
*//*__________________________________________________________________________________________________________________________*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "add_digit.h"

struct BigNumber
{
    BigDigit* digit;
    size_t    length;
};

/*!
@brief  Creates a BigNumber object with digits corresponding to the text

@param  text: string of big number

@return A BigNumber object containing values to the number
*//*__________________________________________________________________________________________________________________________*/
BigNumber* create_BigNumber(const char* text)
{
    size_t len = strlen(text);
    BigNumber* num = (BigNumber*)malloc(sizeof(BigNumber));

    num->length = len;
	num->digit = (BigDigit*)malloc(sizeof(BigDigit) * len);

	for (size_t i = 0; i < len; ++i)
		*(num->digit + i) = (BigDigit)(*(text + i) - '0');

	return num;
}

/*!
@brief  Deallocate memory from the heap

@param  number: reference to the memory address
*//*__________________________________________________________________________________________________________________________*/
void destroy_BigNumber(BigNumber* number)
{
    if(number)
    {
        if(number->digit)
        {
            free(number->digit);
            number->digit = NULL;
        }

        free(number);
        number = NULL;
    }
}

/*!
@brief  Create a new BigNumber object, and sum the two numbers together

@param  number1 & number2: first and second number to be added together

@return BigNumber object that contains the sum of the two number
*//*__________________________________________________________________________________________________________________________*/
BigNumber* add_BigNumber(const BigNumber* number1, const BigNumber* number2)
{
	size_t len = (number1->length > number2->length) ? number1->length : number2->length;

	BigNumber* num = (BigNumber*)malloc(sizeof(BigNumber));
	num->length = len;
	num->digit = (BigDigit*)malloc(sizeof(BigDigit) * len + 1);

	int last1 = (int)number1->length - 1;
	int last2 = (int)number2->length - 1;

	BigDigit carry = 0;

	for (int i = (int)len - 1; i >= 0; --i)
	{
		BigDigit first = (last1 < 0) ? 0 : *(number1->digit + last1--);
		BigDigit second = (last2 < 0) ? 0 : *(number2->digit + last2--);
		*(num->digit + i) = add_BigDigit(first, second, &carry);
	}

	if (*(num->digit) == 0)
	{
		for (size_t i = 0; i < len - 1; ++i)
			*(num->digit + i) = *(num->digit + i + 1);
		if(number1->length != number2->length)
			--num->length;
	}

	return num;
}

/*!
@brief  print appropriate spaces

@param  padding: number of spaces to be printed onto the screen
*//*__________________________________________________________________________________________________________________________*/
void print_space(int padding)
{
	for (int i = 0; i < padding; ++i)
		printf(" ");
}

/*!
@brief  Compare the biggest of two number

@param  a & b: numbers to be compared

@return biggest of the two number
*//*__________________________________________________________________________________________________________________________*/
int max_num(int a, int b)
{
    return a > b ? a : b;
}

/*!
@brief  prints out first and second number to be added, and the sum of these two numbers

@param  number1 & number2: numbers that are added together
        sum: total sum of number1 and number 2
*//*__________________________________________________________________________________________________________________________*/
void print_BigNumber_sum(const BigNumber* number1, const BigNumber* number2, const BigNumber* sum)
{
    int padding = max_num((int)number1->length, max_num((int)number2->length, (int)sum->length)) + 2;

	print_space(padding - (int)number1->length);
	for (size_t i = 0; i < number1->length; ++i)
		printf("%d", *(number1->digit + i));

	printf("\n+");
	print_space(padding - (int)number2->length - 1);
	for (size_t i = 0; i < number2->length; ++i)
		printf("%d", *(number2->digit + i));

	printf("\n");
	for (int i = 0; i < padding; ++i)
		printf("-");

	printf("\n");
	print_space(padding - (int)sum->length);
	for (size_t i = 0; i < sum->length; ++i)
		printf("%d", *(sum->digit + i));
	printf("\n");
}