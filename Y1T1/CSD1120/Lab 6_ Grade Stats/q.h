/*!
@file       q.h
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@tutorial   8
@date       14/10/2021
@brief      function declarations to determine number of students
            enrolled into the course
*//*__________________________________________________________________________*/
#include <stdio.h>

int read_ints_from_stdin(int gr[], int max);
void maximum(int gr[], int tot);
void minimum(int gr[], int tot);
void average(int gr[], int tot);