/*!
@file       main.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@Lab        5
@date       07/10/2021
@brief      handles calculation definition
*//*_______________________________________________________*/
#include <stdio.h>
#include "calc.h"

/*!
@brief      takes in two float and print out the result
            based on the operations
@param      x: first number to be operated on
            y: second number to be operated on
            op: what operations to be done on x and y
*//*_______________________________________________________*/
void calculate(float x, float y, operation op)
{
    switch(op)
    {
        case UNKNOWN:
        {
            printf("\t\tUnknown operation selected!\n");
            break;
        }
        case ADDITION:
        {
            printf("\t\t%f\n", x + y);
            break;
        }
        case SUBTRACTION:
        {
            printf("\t\t%f\n", x - y);            
            break;
        }
        case MULTIPLICATION:
        {
            printf("\t\t%f\n", x * y);
            break;
        }
        case DIVISION:
        {
            if(y != 0)
                printf("\t\t%f\n", x / y);
            else
                printf("\t\tDivision by 0!\n");
            break;
        }
        case DIVISION_INTEGERS:
        {
            if(y != 0)
                printf("\t\t%f\n", (float)((int)x / (int)y));
            else
                printf("\t\tDivision by 0!\n");
            break;
        }
        case MODULUS:
        {
            if(y != 0)
                printf("\t\t%f\n", (float)((int)x % (int)y));
            else
                printf("\t\tDivision by 0!\n");
            break;
        }
    }
}
