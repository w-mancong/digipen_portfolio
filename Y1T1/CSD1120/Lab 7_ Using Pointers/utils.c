/*!
@file       utils.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@assignment 5
@date       07/10/2021
@brief      functions to get used to using pointers
*//*__________________________________________________________________________*/

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

/*!
@brief  get total number of intergers from user
*//*__________________________________________________________________________*/
size_t read_total_count(void)
{
    size_t count;
    printf("Please enter the number of integers: ");
    scanf("%zu", &count);

    if(3 > count)
    {
        printf("There is no third largest number.\n");
        exit (EXIT_FAILURE);
    }
    
    return count;
}

/*!
@brief  read 3 numbers and store them in first, second, third
@param  output parameters
*//*__________________________________________________________________________*/
void read_3_numbers(int* first, int* second, int* third)
{
    scanf("%d %d %d", first, second, third);
}

/*!
@brief  swap the two values
@param  lhs, rhs: values to be swapped
*//*__________________________________________________________________________*/
void swap(int* lhs, int* rhs)
{
    int tmp = *lhs;
    *lhs = *rhs;
    *rhs = tmp;
}

/*!
@brief  get the bigger number from the two
@param  a, b: two numbers to be compared
@return larger number of the two
*//*__________________________________________________________________________*/
int Max2(int a, int b)
{
    return a > b ? a : b;
}

/*!
@brief  sort from decending order (Biggest to smallest)
@param  output parameters
*//*__________________________________________________________________________*/
void sort_3_numbers(int* first, int* second, int* third)
{
    int tmp = Max2(*first, *second);
    if(tmp > *first)
        swap(first, second);

    tmp = Max2(*second, *third);
    if(tmp > *second)
        swap(second, third);

    tmp = Max2(*first, *second);
    if(tmp > *first)
        swap(first, second);
}

/*!
@brief  get a 4th number and rearrange in decending order (first, second, third)
@param  number: possible new biggest number
*//*__________________________________________________________________________*/
void maintain_3_largest(int number, int* first, int* second, int* third)
{
    int tmp1, tmp2;
    if (number > *first)
    {
        tmp1 = *first;
        tmp2 = *second;
        *first = number;
        *second = tmp1;
        *third = tmp2;
    }
    else if (number > *second)
    {
        tmp1 = *second;
        *second = number;
        *third = tmp1;
    }
    else if (number > *third)
    {
        *third = number;
    }
}