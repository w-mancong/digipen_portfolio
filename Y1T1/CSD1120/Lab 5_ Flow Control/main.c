/*!
@file       main.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@Lab        5
@date       07/10/2021
@brief      main function declaration
*//*_______________________________________________________*/

#include <stdio.h>
#include "calc.h"
#include "operation.h"

void ClearBuffer(void)
{

}

/*!
@brief      handles input to calculate
*//*_______________________________________________________*/
int main(void)
{
    // char acceptedSymbols[] = { '+', '-', '*', '/', '\\', '%' };
    
    printf("This program evaluates mathematical expressions.\nThe format of an expression is:\n\tOPERAND1 SYMBOL OPERAND2\nAvailable operation symbols:\n\t+ addition\n\t- subtraction\n\t* multiplication\n\t/ division\n\t\\ integer division\n\t%% modulus\n\n");

    int res = 0; 
    float x = 0.0f, y = 0.0f; 
    char ch = '\0';

    while((res = scanf("%f %c %f", &x, &ch, &y)) && printf("Enter an expression:\n"))
    {                      
        if(res == 3)
        {            
            switch(ch)
            {
                case '+':
                {
                    calculate(x, y, ADDITION);
                    break;    
                }
                case '-':
                {
                    calculate(x, y, SUBTRACTION);
                    break;    
                }
                case '*':
                {
                    calculate(x, y, MULTIPLICATION);
                    break;    
                }
                case '/':
                {
                    calculate(x, y, DIVISION);
                    break;    
                }
                case '\\':
                {
                    calculate(x, y, DIVISION_INTEGERS);
                    break;    
                }
                case '%':
                {
                    calculate(x, y, MODULUS);
                    break;    
                }
                default:
                {
                    calculate(x, y, UNKNOWN);
                    break;    
                }
            }
            int c;
            while((c = getchar()) != '\n' && c != EOF);
        }
        else if(res >= 1)
        {
            //invalid number if arguments
            printf("\t\tInvalid number of arguments!\n");
            int c;
            while((c = getchar()) != '\n' && c != EOF);
            continue;    
        }
        else
        {
            //break out of loop
            break;
        }
    }     
    printf("\tClosing the program...\n");
    return 0;
}
