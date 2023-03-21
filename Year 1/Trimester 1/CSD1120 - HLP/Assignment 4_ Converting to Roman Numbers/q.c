
/*!
@file       q.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@Assignment 4
@date       01/10/2021
@brief      Converts user input decimal value to roman numerals
*//*____________________________________________________________*/
#include "q.h"
#include <stdlib.h>
// #include <string.h>

/*!
@brief  append appropriate letters into the string
@param  
        char* string: passing by reference to print
        out this string
        char c: letter to be appended to the end
        int numeral: how many thousands etc there are
*//*______________________________________________*/
void append_string(const char *c, int numeral)
{
    for(int i = 0; i < numeral; ++i)
        printf("%s", c);
}

/*!
@brief  convert decimal value to roman numerals
@param  
        value : user passed in decimal value
*//*______________________________________________*/
void decimal_to_roman(int value)
{
    // If the number is 4 / 9
    int special = 0;
    
    // 1000
    int one_thousand = value / 1000;
    value -= one_thousand * 1000;
    append_string("M", one_thousand);

    // 900
    special = value / 900;
    value -= special * 900;
    append_string("CM", special);

    // 500
    int five_hundred = value / 500;
    value -= five_hundred * 500;
    append_string( "D", five_hundred);
    
    // 400
    special = value / 400;
    value -= special * 400;
    append_string("CD", special);

    // 100
    int one_hundred = value / 100;
    value -= one_hundred * 100;
    append_string("C", one_hundred);

    // 90
    special = value / 90;
    value -= special * 90;
    append_string("XC", special);

    // 50
    int fifty = value / 50;
    value -= fifty * 50;
    append_string("L", fifty);

    // 40
    special = value / 40;
    value -= special * 40;
    append_string("XL", special);

    // 10
    int ten = value / 10;
    value -= ten * 10;
    append_string("X", ten);

    // 9
    special = value / 9;
    value -= special * 9;
    append_string("IX", special);

    // 5
    int five = value / 5;
    value -= five * 5;
    append_string("V", five);

    // 4
    special = value / 4;
    value -= special * 4;
    append_string("IV", special);
    
    // 1
    append_string("I", value);
    
    printf("\n");
}