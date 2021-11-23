/*!
@file       q.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@tutorial   7
@date       04/11/2021
@brief      Count the total number of latin, control and non latin or control characters
*//*_______________________________________________________________________________________________________*/
#include "q.h"

/*!
@brief  check if current char is upper case letter

@param  ch: integral value of character

@return returns 1 if ch is a upper case letter
*//*_______________________________________________________________________________________________________*/
int is_upper(int ch)
{
    return 'A' <= ch && 'Z' >= ch;
}

/*!
@brief  check if current char is upper case letter

@param  ch: integral value of character

@return returns 1 if ch is a lower case letter
*//*_______________________________________________________________________________________________________*/
int is_lower(int ch)
{
    return 'a' <= ch && 'z' >= ch;
}

/*!
@brief  check if current char is a latin character

@param  ch: integral value of character

@return returns 1 if ch is a latin letter
*//*_______________________________________________________________________________________________________*/
int is_latin(int ch)
{
    return is_upper(ch) || is_lower(ch);
}

/*!
@brief  check if current char is a control character

@param  ch: integral value of character

@return returns 1 if ch is any of the control characters
*//*_______________________________________________________________________________________________________*/
int is_control(int ch)
{
    return '\a' == ch || '\b' == ch ||
        '\f' == ch || '\n' == ch || 
        '\r' == ch || '\t' == ch ||
        '\v' == ch;
}

/*!
@brief  make the output parameters to all zero

@param  latin_freq[]: base address to lain_freq[] array
        size: total elements in latin_freq[] array
        ctrl_cnt: output parameter to store total number of control characters
        non_latin_cnt: output parameter to store total number of non latin and control characters
*//*_______________________________________________________________________________________________________*/
void initialize(int latin_freq[], int size, int *ctrl_cnt, int *non_latin_cnt)
{
    for (int i = 0; i < size; ++i)
        latin_freq[i] = 0;
    *ctrl_cnt = 0;
    *non_latin_cnt = 0;
}

/*!
@brief  based on the character read in the input file, increase the output parameter accordingly

@param  ifs: stream to the file opened
        latin_freq[]: base address to store individual latin characters
        ctrl_cnt: output parameter to store total number of control characters
        non_latin_cnt: output parameter to store total number of non latin and control characters
*//*_______________________________________________________________________________________________________*/
void wc(FILE *ifs, int latin_freq[], int *ctrl_cnt, int *non_latin_cnt)
{
    int ch;
    while((ch = fgetc(ifs)) != EOF)
    {
        if(is_latin(ch))
            ++latin_freq[(is_lower(ch) ? ch - (int)'a' : ch - (int)'A')];
        else if(is_control(ch))
            ++(*ctrl_cnt);
        else
            ++(*non_latin_cnt);
    }
}

/*!
@brief  print out final values of the paramaters

@param  latin_freq[]: base address to the total number of individual latin characters
        size: total elements in latin_freq[] array
        ctrl_cnt: total count to control characters
        non_latin_cnt: total count to neither latin or control characters
*//*_______________________________________________________________________________________________________*/
void print_freqs(int const latin_freq[], int size, int const *ctrl_cnt, int const *non_latin_cnt)
{
    int total_latin_count = 0;
    
    for(int i = 0; i <= size; ++i)
        printf("%*c", (i < size) ? 5 : 0, (i < size) ? (char)((int)'a' + i) : '\n');
    for(int i = 0; i < size; total_latin_count += latin_freq[i], ++i)
        printf("%*d", (i < size) ? 5 : 0, latin_freq[i]);
    fprintf(stdout, "\n");

    const char* str[3] = { "Latin chars", "Non-Latin non-ctrl chars", "Control chars" };
    const int count[3] = { total_latin_count, *non_latin_cnt, *ctrl_cnt };
    for (int i = 0; i < 3; ++i)
        printf("%-24s:%6d\n", str[i], count[i]);
}
