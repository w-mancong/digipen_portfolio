/*!
@file       q.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@tutorial   10
@date       18/11/2021
@brief      functionality to combine file path, to compare, describe and find string
*//*_______________________________________________________________________________________________________*/
#include "q.h"

/*!
@brief  creates a memory address on heap and combine file path using parameter

@param  parent: name of file path
        separator: / for linux // for windows
        folders: an array of different file path names
        count: max number of folders

@return pointer to the first element of memory address created on the heap
*//*_______________________________________________________________________________________________________*/
const char* build_path(const char* parent, const char* separator, const char* const folders[], size_t count)
{
    // additional byte to include null terminator
    size_t size = STRLEN(parent) + 1;
    for(size_t i = 0; i < count; ++i)
    {
        size += STRLEN(*(folders + i)) + STRLEN(separator);
    }
    char* path = (char*)debug_malloc(size);
    // Set all character inside char* to be null terminator
    for(size_t i = 0; i < size; ++i)
        *(path + i) = '\0';    
    STRCPY(path, parent);
    for(size_t i = 0; i < count; ++i)
    {
        STRCAT(path, *(folders + i));
        STRCAT(path, separator);
    }
    return path;
}

/*!
@brief  compares two string

@param  lhs, rhs: pointer to the null terminated byte strings to compare
*//*_______________________________________________________________________________________________________*/
void compare_string(const char* lhs, const char* rhs)
{
    // both strings are equals to each other
    if(STRCMP(lhs, rhs) == 0)
        printf("Both strings are equal.\n");
    else if (STRCMP(lhs, rhs) > 0)
        printf("Right string goes first.\n");
    else if (STRCMP(lhs, rhs) < 0)
        printf("Left string goes first.\n");
}

/*!
@brief  describe the length and path of the text

@param  text: path name
*//*_______________________________________________________________________________________________________*/
void describe_string(const char* text)
{
    printf("The length of the path \"%s\" is %zu.\n", text, STRLEN(text));
}

/*!
@brief  Finds string inside a string

@param  string: pointer to the null-terminated byte string to examine
        substring: pointer to the null-terminated byte string to search for
*//*_______________________________________________________________________________________________________*/
void find_string(const char* string, const char* substring)
{
#define TEXT_SIZE   2
    const char* print_text[TEXT_SIZE] = { "Text:", "Sub-text:" };
    const char* param_text[TEXT_SIZE] = { string, substring };
    
    printf("Searching for a string:\n");
    
    for(int i = 0; i < TEXT_SIZE; ++i)
        printf("\t%-10s%s\n", *(print_text + i), *(param_text + i));

    char *pos = STRSTR(string, substring);
    if(pos)
    {
        printf("\t%-10sfound %zu characters at a position %ld.\n", "Result:", STRLEN(substring), pos - string);
    }
    else
        printf("\t%-10s%s\n", "Result:", "not found");
}
