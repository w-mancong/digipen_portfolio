/*!*****************************************************************************
\file linearSearch.c
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: Operating System
\par Quiz 3
\date 14-10-2022
\brief
This file contains function declarations that does a linear search using
multiple threads
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
 
// Max size of array
#define MAX_SIZE 100000
 
// Number of threads to create
#define threadSize 4

int arr[MAX_SIZE];
int key;

// Flag to indicate if key is found in arr[].
int findFlag = 0;

typedef struct SearchStruct SearchStruct;

/*!*****************************************************************************
    \brief Struct to contain pointer to a half-open array of arr
*******************************************************************************/
struct SearchStruct
{
    int *beg;   // Pointer to the first element to search
    int *end;   // Pointer to the last element
} search[threadSize];

/*!*****************************************************************************
    \brief Function that does linear search

    \param [in] args: Pointer to search
*******************************************************************************/
void* TLinearSearch(void* args)
{
    SearchStruct ss = *(SearchStruct *)args;

    for (int *ptr = ss.beg; ptr < ss.end && !findFlag; ++ptr)
    {
        if(*ptr == key)
            findFlag = 1;
    }

    pthread_exit((void *)NULL);
}
 

// Linear search key in the values 
int main(int argc, char **argv)
{
    // Store the thread_ids
    pthread_t threads[threadSize];

    // Get the input
    // The first input is the key
    scanf("%d", &key);

    // The rest are the the integer values
    int index = 0;
    int value = 0;
    memset(arr, 0, sizeof(arr));
    while (!feof(stdin))
    {
        scanf("%d", &value);
        arr[index] = value;
        index++;
    }
    int arrSize = index;

    size_t totalThreads = threadSize;
    if(arrSize < threadSize)
        totalThreads = arrSize % threadSize;

    memset(search, 0, sizeof(SearchStruct) * threadSize);
    size_t increment = (arrSize / totalThreads);
    // Need to split evenly
    if (arrSize % totalThreads)
    {
        increment += 1;
        for (size_t i = 0, beg = 0, end = increment, counter = arrSize; i < totalThreads; ++i, beg += increment, end += increment, counter -= increment)
        {
            if(counter < increment)
                end = beg + counter;
            (search + i)->beg = arr + beg;
            (search + i)->end = arr + end;
        }
    }
    // Equal splits
    else
    {
        for (size_t i = 0, beg = 0, end = increment; i < totalThreads; ++i, beg += increment, end += increment)
        {
            (search + i)->beg = arr + beg;
            (search + i)->end = arr + end;
        }
    }
    
    for(size_t i = 0; i < totalThreads; ++i)
    {
        if (pthread_create(threads + i, NULL, &TLinearSearch, search + i))
            perror("Error: Failed to create thread\n");
    }
    
    for (size_t i = 0; i < totalThreads; ++i)
    {
        if(pthread_join(*(threads + i), NULL))
            perror("Error: Failed to join thread\n");
    }

    // Print the result
    if (findFlag == 1)
        printf("Found\n");
    else
        printf("Not found\n");

    return 0;
}