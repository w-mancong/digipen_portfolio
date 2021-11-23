/*!
@file       q.h
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@tutorial   7
@date       04/11/2021
@brief      Count the total number of latin, control and non latin or control characters
*//*_______________________________________________________________________________________________________*/
#include <stdio.h>

/*!
@brief  make the output parameters to all zero

@param  latin_freq[]: base address to lain_freq[] array
        size: total elements in latin_freq[] array
        ctrl_cnt: output parameter to store total number of control characters
        non_latin_cnt: output parameter to store total number of non latin and control characters
*//*_______________________________________________________________________________________________________*/
void initialize(int latin_freq[], int size, int *ctrl_cnt, int *non_latin_cnt);

/*!
@brief  based on the character read in the input file, increase the output parameter accordingly

@param  ifs: stream to the file opened
        latin_freq[]: base address to store individual latin characters
        ctrl_cnt: output parameter to store total number of control characters
        non_latin_cnt: output parameter to store total number of non latin and control characters
*//*_______________________________________________________________________________________________________*/
void wc(FILE *ifs, int latin_freq[], int *ctrl_cnt, int *non_latin_cnt);

/*!
@brief  print out final values of the paramaters

@param  latin_freq[]: base address to the total number of individual latin characters
        size: total elements in latin_freq[] array
        ctrl_cnt: total count to control characters
        non_latin_cnt: total count to neither latin or control characters
*//*_______________________________________________________________________________________________________*/
void print_freqs(int const latin_freq[], int size, int const *ctrl_cnt, int const *non_latin_cnt);