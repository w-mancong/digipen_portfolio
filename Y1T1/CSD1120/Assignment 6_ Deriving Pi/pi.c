/*!
@file       pi.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@assignment 5
@date       07/10/2021
@brief      function definitions to compute the approximate value of pi
*//*__________________________________________________________________________*/
#include "pi.h"

#define RADIUS 2.0

/*!
@brief  approximately find the value of pi based on the number rect
@param  number of rectangles
@return approximate value of pi
*//*_________________________________________________________________________*/
double calculus_pi(int slices)
{
    double width = RADIUS / slices;
    double mid_point = width / 2.0;
    double area = 0.0;

    for(int i = 0; i < slices; ++i)
    {
        double x = mid_point + width * i; 
        double height = sqrt(RADIUS * RADIUS - x * x);
        area += width * height;
    }
    
    return area;
}

/*!
@brief  the summation of infinite series of additions and subtractions
@param  number of times to iterate through
@return approximate value of pi
*//*_________________________________________________________________________*/
double leibniz_pi(int terms)
{
    double pi = 1.0;
    double negate = -1.0;
    const double dividen = 3.0;
    
    for (int i = 0; i < terms - 1; ++i)
    {
        pi += 1.0 / (dividen + (i * 2.0)) * negate;
        negate = -negate;
    }

    return pi * 4.0;
}