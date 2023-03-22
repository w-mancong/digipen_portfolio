/*!
@file       main.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@assignment 7
@date       29/10/2021
@brief      function definitions to Vigenère cipher algorithm
*//*__________________________________________________________________________*/

#include <stdio.h>			// file and console I/O

#define INCLUDE_KEYWORD
#include "q.h"				// include declarations and the decryption keyword

/*!
@brief	Check if current character is a whitespace

@param	c: character to be checked

@return	returns 1 if the current character is a whitespace
*//*__________________________________________________________________________*/
unsigned char IsWhiteSpaces(char c)
{
	return c == ' ' || c == '\n' ||
		   c == '\r' || c == '\t';
}

/*!
@brief	main entry for the program
*//*__________________________________________________________________________*/
int main(void)
{	
	const unsigned size = (unsigned)(sizeof(key) / sizeof(key[0]));
	unsigned keyIndex = 0;
	
	#ifdef ENCRYPT	
	// TODO: encrypt characters from plain.txt and put them in file cipher.txt
	FILE* rFile = fopen("plain.txt", "r");
	while(!feof(rFile))
	{
		char cArr[2] = "";

		// get single char from file
		cArr[0] = (char)fgetc(rFile);

		// if it's EOF char, don't convert
		if(cArr[0] == EOF)
			break;		

		// encrypt char here
		encrypt(cArr, key[keyIndex++]);
		
		// append into cipher.txt
		FILE* aFile = fopen("cipher.txt", "a");
		fprintf(aFile, "%c", cArr[0]);
		fclose(aFile);
		
		// if out of bound, reset it to 0
		if(size <= keyIndex)
			keyIndex = 0;
	}
	fclose(rFile);	
	#else
	// TODO: decrypt cipher.txt into out_plain.txt
	FILE* rFile = fopen("cipher.txt", "r");
	unsigned count = 0;
	unsigned char gotWords = 0;
	while(!feof(rFile))
	{
		char cArr[2] = "";
		// get single char from file
		cArr[0] = (char)fgetc(rFile);

		// if it's EOF char, don't convert
		if(cArr[0] == EOF)
			break;		

		// encrypt char here
		decrypt(cArr, key[keyIndex++]);
		
		if(!gotWords && !IsWhiteSpaces(cArr[0]))
		{
			gotWords = 1;
		}
		else if(gotWords && IsWhiteSpaces(cArr[0]))
		{
			gotWords = 0;
			++count;
		}	
		
		// append into cipher.txt
		FILE* aFile = fopen("out_plain.txt", "a");	
		fprintf(aFile, "%c", cArr[0]);
		fclose(aFile);
		
		// if out of bound, reset it to 0
		if(size <= keyIndex)
			keyIndex = 0;
	}
	fclose(rFile);	
	// TODO: write count of words into stderr
	fprintf(stderr, "Words: %u\n", count);

	#endif

	return 0;
}
