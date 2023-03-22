/*!
@file       q.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@assignment 7
@date       29/10/2021
@brief      function definitions to Vigenère cipher algorithm
*//*__________________________________________________________________________*/

#include "q.h"

/*!
@brief	encrypting using the key

@param  letter: current letter to be encrypting
        key: special key for the letter to be encrypting
*//*__________________________________________________________________________*/
void encrypt(char* letter, char key)
{
    int flag = (int)letter[0] + (int)key;
    if(ASCII_COUNT <= flag)
    {
        flag -= ASCII_COUNT;
        letter[0] = (char)flag;
    }
    else
    {
        letter[0] = (char)((int)letter[0] + (int)key);
    }
}

/*!
@brief	decrypting using the key

@param  letter: current letter to be decrypted
        key: special key for the letter to be decrypted
*//*__________________________________________________________________________*/
void decrypt(char* letter, char key)
{
    int flag = (int)letter[0] - (int)key;
    if(0 > flag)
    {
        flag += ASCII_COUNT;
        letter[0] = (char)flag;
    }
    else
    {
        letter[0] = (char)((int)letter[0] - (int)key);
    }
}