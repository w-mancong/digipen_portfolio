/*!
@file       q.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@tutorial   8
@date       14/10/2021
@brief      function definitions to determine number of students
            enrolled into the course
*//*__________________________________________________________________________*/
#include "q.h"

/*!
@brief      read input from file and store it into an array

@param      gr is an array that stores the score of each individual student
            max is the maximum number of students and size of the array

@return     returns the total number int read into the array
*//*__________________________________________________________________________*/
int read_ints_from_stdin(int gr[], int max)
{
    int i = 0;
    for(int num; (scanf("%d", &num) != EOF) && i < max; gr[i++] = num)
    {
        // empty by design
    }
    return i;
}

/*!
@brief      find the largest score in the array

@param      gr is the array of score of each individual students
            tot is the total number of students
*//*__________________________________________________________________________*/
void maximum(int gr[], int tot)
{
    int max = gr[0];
    
    for(int i = 1; i < tot; max = (max < gr[i] ? gr[i] : max), ++i)
    {
        // empty by design
    }    
    printf("Max of %d is %d\n", tot, max);
}

/*!
@brief      find the minimm score in the array

@param      gr is the array of score of each individual students
            tot is the total number of students
*//*__________________________________________________________________________*/
void minimum(int gr[], int tot)
{
    int min = gr[0];
    
    for(int i = 1; i < tot; min = (min > gr[i] ? gr[i] : min), ++i)
    {
        // empty by design
    }
    printf("Min of %d is %d\n", tot, min);
}

/*!
@brief      find the average score in the array

@param      gr is the array of score of each individual students
            tot is the total number of students
*//*__________________________________________________________________________*/
void average(int gr[], int tot)
{
    double avg = 0;
    
    for(int i = 0; i < tot; avg += gr[i++])
    {
        // empty by designs
    }    
    printf("Avg of %d is %.2f\n", tot, avg /= tot);
}