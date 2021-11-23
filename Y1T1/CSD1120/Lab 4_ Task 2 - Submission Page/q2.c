/*!
@file       q.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@Lab        4
@date       30/09/2021
@brief      Convert Singapore denomination into different
            Singapore notes and coins
*//*____________________________________________________________*/

#include <stdio.h>

int index = 0;

/*!
@brief  check if it's valid amount of notes
@param  denomination: amount entered by user
*//*____________________________________________*/
int valid_notes(int denomination)
{
    if(denomination < 0)
    {
        return 0;
    }
    return 1;
}

/*!
@brief  check if it's valid amount of coins
@param  cents: amount entered by user
*//*____________________________________________*/
int valid_coins(int cents)
{
    if(cents < 0 || cents >= 100)
    {
        return 0;
    }
    return 1;
}

/*!
@brief  print header of the table
*//*____________________________________________*/
void print_header(void)
{
    printf("\n+----+--------------+-------+\n| #  | Denomination | Count |\n+----+--------------+-------+\n");
}

/*!
@brief  print 
@param  index: first column
        cents: broken down counter
        value: denomination of the cents
*//*____________________________________________*/
void print_line(int index, int cents, int value)
{
    int notes = value / 100;
    int coins = (int)((((float)value / 100.0f) - (float)notes) * 100);

    printf("| %-3d|%10d.%.2d |%6d |\n", index, notes, coins, cents);
}

/*!
@brief  break down coins then prints header of output 
        table by calling print_line()
@param breakdown into coins
*//*____________________________________________*/
void coins(int cents)
{
    int hundred_dollar = 0, fifty_dollar = 0, ten_dollar = 0, five_dollar = 0, two_dollar = 0, one_dollar = 0, fifty_cents = 0, twenty_cents = 0, ten_cents = 0, five_cents = 0, one_cent = 0;
    int remainder = cents;
    int index = 0;
    
    print_header();

    hundred_dollar = remainder / 10000;
    remainder -= hundred_dollar * 10000;
    print_line(++index, hundred_dollar, 10000);

    fifty_dollar = remainder / 5000;
    remainder -= fifty_dollar * 5000;
    print_line(++index, fifty_dollar, 5000);
    
    ten_dollar = remainder / 1000;
    remainder -= ten_dollar * 1000;   
    print_line(++index, ten_dollar, 1000);

    five_dollar = remainder / 500;
    remainder -= five_dollar * 500;
    print_line(++index, five_dollar, 500);
    
    two_dollar = remainder / 200;
    remainder -= two_dollar * 200;
    print_line(++index, two_dollar, 200);
    
    one_dollar = remainder / 100;
    remainder -= one_dollar * 100;
    print_line(++index, one_dollar, 100);

    fifty_cents = remainder / 50;
    remainder -= fifty_cents * 50;
    print_line(++index, fifty_cents, 50);
    
    twenty_cents = remainder / 20;
    remainder -= twenty_cents * 20;
    print_line(++index, twenty_cents, 20);
    
    ten_cents = remainder / 10;
    remainder -= ten_cents * 10;
    print_line(++index, ten_cents, 10);
    
    five_cents = remainder / 5;
    remainder -= five_cents * 5;
    print_line(++index, five_cents, 5);
    
    one_cent = remainder / 1;
    remainder -= one_cent * 1;
    print_line(++index, one_cent, 1);
}

/*!
@brief main entry for the program
*//*____________________________________________*/
int main(void)
{
    int flag = 0, denomination = 0, cents = 0;

    while (1)
    {
        printf("Please enter total value: ");  
        flag = scanf("%d.%d", &denomination, &cents);

        if(flag != 2)    
            break;

        if(!valid_notes(denomination) || !valid_coins(cents))
            break;

        coins((denomination * 100) + cents);
        printf("+----+--------------+-------+\n");
    }
    printf("Program ended\n");
}
