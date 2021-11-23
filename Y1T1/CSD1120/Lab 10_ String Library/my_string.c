/*!
@file       my_string.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@tutorial   10
@date       18/11/2021
@brief      my own implementation of string.h 
*//*_______________________________________________________________________________________________________*/
#include "my_string.h"

/*!
@brief  count the length of the string not including null terminator

@param  str: reference to the string

@return length of the current string
*//*_______________________________________________________________________________________________________*/
size_t my_strlen(const char* str)
{
    const char* ptr = str;
    while (*(++str));
    return (size_t)(str - ptr);
}

/*!
@brief  copies from string from src into dest

@param  dest: pointer to the array to write to
        src: pointer to the null terminated byte string to copy from

@return pointer to the first element of dest
*//*_______________________________________________________________________________________________________*/
char* my_strcpy(char* dest, const char* src)
{
    size_t size = my_strlen(src);
    for (size_t i = 0; i < size; ++i)
        *(dest + i) = *(src + i);
    return dest;
}

/*!
@brief  appends a copy of the null terminated byte string from src into dest

@param  dest: pointer to the null terminated byte string to append to
        src: pointer to the null terminated byte string to append from

@return pointer to the first element of dest
*//*_______________________________________________________________________________________________________*/
char* my_strcat(char* dest, const char* src)
{    
    while (*(++dest));
    my_strcpy(dest, src);
    return dest;
}

/*!
@brief  compares two null terminated byte string lexicographically

@param  lhs, rhs: pointer to the null terminated byte strings to compare

@return Negative value if lhs appears before rhs in lexicographically
        Zero if lhs and rhs are the same
        Positive value if lhs appears after rhs in lexicographically
*//*_______________________________________________________________________________________________________*/
int my_strcmp(const char* lhs, const char* rhs)
{
    do
    {
        if (*lhs > *rhs)
            return 1;
        else if (*lhs < *rhs)
            return -1;
    } while (*lhs++ && *rhs++); 
    return 0;
}

/*!
@brief  Finds the first occurrence of the null-terminated byte string pointed to by substr in the 
        null-terminated byte string pointed to by str. The terminating null characters are not compared.
        The behavior is undefined if either str or substr is not a pointer to a null-terminated byte string.

@param  str: pointer to the null-terminated byte string to examine
        substr: pointer to the null-terminated byte string to search for

@return Pointer to the first character of the found substring in str, or a null pointer 
        if such substring is not found. If substr points to an empty string, str is returned.
*//*_______________________________________________________________________________________________________*/
char* my_strstr(const char* str, const char* substr)
{
    if (!my_strlen(substr))
        return (char*)str;

    size_t str_len = my_strlen(str);
    size_t substr_len = my_strlen(substr);

    const char* ptr = str;
    for (size_t i = 0; i + substr_len <= str_len; ++i)
    {
        if (*(ptr + i) != *substr)
            continue;

        const char* str_tmp = ptr + i;
        size_t counter = 1;
        for (size_t j = 1; j < substr_len; ++j)
        {
            if (*(substr + j) != *(str_tmp + j))
                break;
            ++counter;
        }
        if (counter == substr_len)
            return (char*)str_tmp;
    }
    return NULL;
}