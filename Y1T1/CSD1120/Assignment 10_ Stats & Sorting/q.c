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
*//*____________________________________________________________________________________________*/
#include "q.h"

/*!
@brief	read and store individual data from txt file

@param  file_name: name of the .txt file
        ptr_cnt: output parameter to store the total number of data in txt file

@return pointer to the address of the first element with it's memory allocated in the heap
*//*____________________________________________________________________________________________*/
double* read_data(char const *file_name, int *ptr_cnt)
{
    FILE* stream = fopen(file_name, "r");
    if (!stream)
        return NULL;

#define BUFFER_SIZE 1024
    double* total = NULL;
    int size = 0, curr_index = 0;
    char buffer[BUFFER_SIZE];
    while (fgets(buffer, BUFFER_SIZE, stream))
    {
        // loop through to find how many grades are in the file
        for (int i = 0; *(buffer + i) != '\0'; ++i)
        {
            if (*(buffer + i) != ' ' && *(buffer + i) != '\n')
                continue;
            ++size;
        }
        total = (double*)realloc((total != NULL ? total : NULL), sizeof(double) * size);
        char dbl_string[6] = "";
        int dbl_index = 0;
        for (int i = 0; *(buffer + i) != '\0'; ++i)
        {
            if (*(buffer + i) != ' ' && *(buffer + i) != '\n')
                *(dbl_string + dbl_index++) = *(buffer + i);
            else
            {
                *(total + curr_index++) = atof(dbl_string);
                dbl_index = 0;
            }
        }
    }
    fclose(stream);

    *ptr_cnt = size;
    return total;
}

/*!
@brief	find the largest number within the half open range array

@param  begin: pointer to the first element
        end: pointer to the last element
        
@return largest number in the half open range array
*//*____________________________________________________________________________________________*/
double max(double const *begin, double const *end)
{
    double largest = *begin;
    for (int i = 1; begin + i < end; ++i)
    {
        if (*(begin + i) > largest)
            largest = *(begin + i);
    }
    return largest;
}

/*!
@brief	find the smallest number within the half open range array

@param  begin: pointer to the first element
        end: pointer to the last element
        
@return largest smallest in the half open range array
*//*____________________________________________________________________________________________*/
double min(double const *begin, double const *end)
{
    double smallest = *begin;
    for (int i = 1; begin + i < end; ++i)
    {
        if(*(begin + i) < smallest)
            smallest = *(begin + i);
    }
    return smallest;
}

/*!
@brief	summation of all the values inside the half open range array and divide by the total 
        number of elements inside this half open range array

@param  begin: pointer to the first element
        end: pointer to the last element
        
@return the mean of value of all the values in the half open range 
*//*____________________________________________________________________________________________*/
double average(double const *begin, double const *end)
{
    double sum = *begin;
    for (int i = 1; begin + i < end; ++i)
        sum += *(begin + i);
    return sum / (double)(end - begin);
}

/*!
@brief	find the variance of all the elements inside the half open range array

@param  begin: pointer to the first element
        end: pointer to the last element
        
@return sample of variance
*//*____________________________________________________________________________________________*/
double variance(double const *begin, double const *end)
{
    double mean = average(begin, end), sum = 0;
    for (int i = 0; begin + i < end; ++i)
    {
        double var = *(begin + i) - mean;
        sum += var * var;
    }
    return sum / ((double)(end - begin) - 1.0);
}

/*!
@brief	find the standard deviation of all the elements inside the half open range array

@param  begin: pointer to the first element
        end: pointer to the last element
        
@return the standard deviation of the data inside the half open range array
*//*____________________________________________________________________________________________*/
double std_dev(double const *begin, double const *end)
{
    return sqrt(variance(begin, end));
}

/*!
@brief	find the median from the data set

@param  base: pointer to the first element
        size: total elements inside the half open range array
        
@return median of all the values
*//*____________________________________________________________________________________________*/
double median(double *base, int size)
{
    selection_sort(base, size); 
    if(size % 2) // odd numbers
        return *(base + (size >> 1));
    else
        return (*(base + ((size >> 1) - 1)) + *(base + (size >> 1))) * 0.5;
}

/*!
@brief	does a simple sorting algorithm from smallest to biggest

@param  base: pointer to the first element
        size: total elements inside the half open range array
*//*____________________________________________________________________________________________*/
void selection_sort(double *base, int size)
{
    for (int i = 0; i < size - 1; ++i)
    {
        int index = i;
        double smallest = *(base + i);
        for (int j = i + 1; j < size; ++j)
        {
            // find the smallest index
            if (*(base + j) < smallest)
            {
                smallest = *(base + j);
                index = j;
            }
        }
        if (*(base + i) != *(base + index))
        {
            double tmp = *(base + i);
            *(base + i) = *(base + index);
            *(base + index) = tmp;
        }
    }
}

/*!
@brief	store the total percentage of grades from the data set

@param  begin: pointer to the first element
        end: pointer to the last element
        ltr_grades: output parameter to store the percentage of grades
*//*____________________________________________________________________________________________*/
void ltr_grade_pctg(double const *begin, double const *end, double *ltr_grades)
{
    int list_grades[5] = { 0 };
    for (int i = 0; begin + i < end; ++i)
    {
        if (*(begin + i) >= 90.0)
            ++*(list_grades + A_GRADE);
        else if (*(begin + i) >= 80.0 && *(begin + i) < 90.0)
            ++*(list_grades + B_GRADE);
        else if (*(begin + i) >= 70.0 && *(begin + i) < 80.0)
            ++*(list_grades + C_GRADE);
        else if (*(begin + i) >= 60.0 && *(begin + i) < 70.0)
            ++*(list_grades + D_GRADE);
        else if (*(begin + i) < 60.0)
            ++*(list_grades + F_GRADE); 
    }
    double percent = (1.0 / (double)(end - begin)) * 100.0;
    for (int i = 0; i < TOT_GRADE; ++i)
        *(ltr_grades + i) = *(list_grades + i) * percent;
}
