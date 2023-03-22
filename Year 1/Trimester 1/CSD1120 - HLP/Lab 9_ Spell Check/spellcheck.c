/*!
@file       spellcheck.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@tutorial   9
@date       11/11/2021
@brief      does a simple spell checking algorithm 
*//*__________________________________________________________________________________________________*/

#include <string.h>		// strcpy, strcmp, strlen                                   
#include <stdio.h>		// printf, fopen, fclose

#include "spellcheck.h"

/*!
@brief  remove first instance of newline character

@param  str: reference to the string to have it's newline character removed

@return pointer to the first element in str
*//*_______________________________________________________________________________________________________*/
char* remove_newline_seq(char* str)
{
	for (int i = 0; str[i] != '\0'; ++i)
	{
		if ('\n' == str[i] || '\r' == str[i])
		{
			str[i] = '\0';
			return str;
		}
	}
    return NULL;
}

/*!
@brief  turn the all the characters in the string from lower case to upper case characters

@param  string: string that will be converted to upper case characters

@return pointer to the first element in string
*//*_______________________________________________________________________________________________________*/
char* str_to_upper(char* string)
{
	for (int i = 0; string[i] != '\0'; ++i)
	{
		if ('a' <= string[i] && 'z' >= string[i])
			string[i] = (char)((int)string[i] - ((int)'a' - (int)'A'));
	}
	return string;
}

/*!
@brief  get the total number of words that start with character 'letter'

@param  dictionary: file path to the text file to be opened
        letter: letter to be checked
            
@return the total number of words that start with letter
*//*_______________________________________________________________________________________________________*/
WordCount words_starting_with(const char* dictionary, char letter)
{
    FILE* stream = fopen(dictionary, "r");
    if(!stream)
        return FILE_ERR_OPEN;
        
    WordCount count = 0;
    char buffer[LONGEST_WORD + 1];
    while(fgets(buffer, LONGEST_WORD + 1, stream))
    {
        str_to_upper(buffer);
        if('a' <= letter && 'z' >= letter)
            letter = (char)((int)letter - ((int)'a' - (int)'A'));
        if(buffer[0] != letter)
            continue;
        ++count;
    }    
    return count;
}

/*!
@brief  Look up the word in dictionary to check if the spelling is correct

@param  dictionary: file path to the text file to be opened
        word: word to be checked

@return WORD_OK if word is found and spelling is correct, else WORD_BAD if no word is found in dictionary
*//*_______________________________________________________________________________________________________*/
ErrorCode spell_check(const char* dictionary, const char* word)
{
	FILE* stream = fopen(dictionary, "r");
	if (!stream)
		return FILE_ERR_OPEN;

	char buffer[LONGEST_WORD + 1];
	char cmpstr[LONGEST_WORD + 1];
	strcpy(cmpstr, word);
	while (fgets(buffer, LONGEST_WORD + 1, stream))
	{
		remove_newline_seq(buffer);
		str_to_upper(buffer); str_to_upper(cmpstr);
		if (!strcmp(buffer, cmpstr))
			return WORD_OK;
	}
	return WORD_BAD;
}

/*!
@brief  Store the number of words with length 'count' and storing it in corresponding position in 'lengths'

@param  dictionary: file path to the text file to be opened
        lengths: array that stores the total number of words with length 'count'
        count: maximum length of the word

@return FILE_OK if the file can be opened successfully
*//*_______________________________________________________________________________________________________*/
ErrorCode word_lengths(const char* dictionary, WordCount lengths[], WordLength count)
{
    FILE* stream = fopen(dictionary, "r");
    if(!stream)
        return FILE_ERR_OPEN;
    
    char buffer[LONGEST_WORD + 1];
    while(fgets(buffer, LONGEST_WORD + 1, stream))
    {
        remove_newline_seq(buffer);
        int len = (int)strlen(buffer);
        if(len > count || 0 >= len)
            continue;
        ++lengths[len];
    }   
    return FILE_OK;
}

/*!
@brief  store the total number of word counts and word length in struct DictionaryInfo

@param  dictionary: file path to the text file to be opened
        info: storing corresponding data in struct DictionaryInfo

@return FILE_OK if the file can be opened successfully
*//*_______________________________________________________________________________________________________*/
ErrorCode info(const char* dictionary, DictionaryInfo* info)
{
    FILE* stream = fopen(dictionary, "r");
    if(!stream)
        return FILE_ERR_OPEN;
    
    char buffer[LONGEST_WORD + 1];
    info->count = 0; info->longest = 0; info->shortest = LONGEST_WORD + 1;
    while(fgets(buffer, LONGEST_WORD + 1, stream))
    {
        remove_newline_seq(buffer);
        WordLength len = (WordLength)strlen(buffer);
        if(len < info->shortest)
            info->shortest = len;
        if(len > info->longest)
            info->longest = len;        
        ++info->count;
    }   
    return FILE_OK;
}
