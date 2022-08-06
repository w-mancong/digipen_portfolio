/*!
@file       q.c 
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@assignment 3
@date       23/9/2021
@brief      This file contains basic functions to convert from Fahrenheit to
            Celsius and Kevin
*//*_________________________________________________________________________*/

#include <stdio.h>

void temperature_convertor(int fahrenheit)
{
	double celsius = (double)((fahrenheit - 32) * 5) / 9;
    double kevin = celsius + 273.15;

    printf("Fahrenheit     Celsius        Kelvin         \n---------------------------------------------\n%-15d%-15.2f%-15.2f\n", fahrenheit, celsius, kevin);
}