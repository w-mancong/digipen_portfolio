/*!
@file       q.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@assignment 10
@date       20/11/2021
@brief      A report that provides the minimum, maximum, average, and median grades, 
            variance and standard deviation of the grades, and a table indicating 
            the percentage of grades in each letter grade category
*//*_______________________________________________________________________________________*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef enum Grades
{
    A_GRADE,
    B_GRADE,
    C_GRADE,
    D_GRADE,
    F_GRADE,
    TOT_GRADE,
} Grades;

/*!
@brief	read and store individual data from txt file

@param  file_name: name of the .txt file
        ptr_cnt: output parameter to store the total number of data in txt file

@return pointer to the address of the first element with it's memory allocated in the heap
*//*____________________________________________________________________________________________*/
double* read_data(char const *file_name, int *ptr_cnt);

/*!
@brief	find the largest number within the half open range array

@param  begin: pointer to the first element
        end: pointer to the last element
        
@return largest number in the half open range array
*//*____________________________________________________________________________________________*/
double max(double const *begin, double const *end);

/*!
@brief	find the smallest number within the half open range array

@param  begin: pointer to the first element
        end: pointer to the last element
        
@return largest smallest in the half open range array
*//*____________________________________________________________________________________________*/
double min(double const *begin, double const *end);

/*!
@brief	summation of all the values inside the half open range array and divide by the total 
        number of elements inside this half open range array

@param  begin: pointer to the first element
        end: pointer to the last element
        
@return the mean of value of all the values in the half open range 
*//*____________________________________________________________________________________________*/
double average(double const *begin, double const *end);

/*!
@brief	find the variance of all the elements inside the half open range array

@param  begin: pointer to the first element
        end: pointer to the last element
        
@return sample of variance
*//*____________________________________________________________________________________________*/
double variance(double const *begin, double const *end);

/*!
@brief	find the standard deviation of all the elements inside the half open range array

@param  begin: pointer to the first element
        end: pointer to the last element
        
@return the standard deviation of the data inside the half open range array
*//*____________________________________________________________________________________________*/
double std_dev(double const *begin, double const *end);

/*!
@brief	find the median from the data set

@param  base: pointer to the first element
        size: total elements inside the half open range array
        
@return median of all the values
*//*____________________________________________________________________________________________*/
double median(double *base, int size);

/*!
@brief	does a simple sorting algorithm from smallest to biggest

@param  base: pointer to the first element
        size: total elements inside the half open range array
*//*____________________________________________________________________________________________*/
void selection_sort(double *base, int size);

/*!
@brief	store the total percentage of grades from the data set

@param  begin: pointer to the first element
        end: pointer to the last element
        ltr_grades: output parameter to store the percentage of grades
*//*____________________________________________________________________________________________*/
void ltr_grade_pctg(double const *begin, double const *end, double *ltr_grades);
