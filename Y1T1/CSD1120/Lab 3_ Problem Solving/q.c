/*!
@file       q.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@tutorial   3
@date       23/09/2021
@brief      This file contains function to simulate the change 
            dispenser in a vending machine
*//*____________________________________________________________*/

#include <stdio.h>

/*!
@brief
    calculate amount of change to return
@param      
    change: current change
    amount: which combination of coins is this?
            100 -> dollars / loonies
            50  -> 50 cents / half-loonies
            25  -> 25 cents / quarters
            10  -> 10 cents / dimes
            5   -> 5 cents / nickles
            1   -> 1 cent / penny
*//*_____________________________________________________________*/
int calculate_change(int* change, int amount)
{
    // total: how many loonies, half-loonies..
    // temp_change: temporary change that will hold and calculate the change
    int total = 0, temp_change = *change;
    while(1)
    {
        temp_change -= amount;
        if(temp_change < 0)
        {
            *change = temp_change + amount;
            return total;
        }       
        ++total;
    }
}

/*!
@brief
    dispenses the right amount of change to the buyer
@param      
    denomination: total amount paid by user in dollars
    price_in_cents: price of items in cents
*//*_____________________________________________________________*/
void dispense_change(int denomination, int price_in_cents)
{    
    // get change in terms of cents
    int change = (denomination * 100) - price_in_cents;
    int loonies = 0, half_loonies = 0, quarters = 0, dimes = 0, nickels = 0, pennies = 0;
    
    if(change < 0)
    {
        printf("You do not have enough money\n");
        return;
    }

    loonies = calculate_change(&change, 100);
    half_loonies = calculate_change(&change, 50);
    quarters = calculate_change(&change, 25);
    dimes = calculate_change(&change, 10);
    nickels = calculate_change(&change, 5);
    pennies = calculate_change(&change, 1);

    printf("%d loonies + %d half-loonies + %d quarters + %d dimes + %d nickels + %d pennies\n", loonies, half_loonies, quarters, dimes, nickels, pennies);
}
