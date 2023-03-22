/*! 
@file       calc.h
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD1120
@section    B
@tutorial   Tutorial 2
@date       16/09/2021
@brief      This file contains function definitions for simple
            math calculations
*//*_______________________________________________________________*/

#include "calc.h"

// finds the square of x
int sqaure(int x)
{
    return x * x;
}

// finds the cube of x
double cube(double x)
{
    return x * x * x;
}

// returns the negative value of x
double minus(double x)
{
    return -x;
}