/*!
@file       Provide function definitions to draw tree & animal
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@tutorial   Assignment: Introduction to C
@date       16/09/2021
@brief      This file contains function definition to draw out the animal and
            tree and prints it out for the user to see
*//*_________________________________________________________________________________*/

// void draw_tree(void);
// void draw_animal(void);

#include <stdio.h>

/*!
@brief drawing a tree using * and #

    *
   ***
  *****
 *******
*********
    #
    #
    #
    #

Draws out a tree
*//*______________________________________________*/
void draw_tree(void)
{
    printf("    *\n   ***\n  *****\n *******\n*********\n    #\n    #\n    #\n    #\n");
}

/*!
@brief drawing an animal using /'\_()

  /\     /\
 /  \___/  \
(           )    -------
(   '   '   )   / Hello  \
(     _     )  <  Junior  |
(           )   \ Coder! /
 |         |     -------
 |    |    |
 |    |    |
(_____|_____)

Draws out an animal
*//*______________________________________________*/
void draw_animal(void)
{
    printf("  /\\     /\\\n /  \\___/  \\\n(           )    -------\n(   '   '   )   / Hello  \\\n(     _     )  <  Junior  |\n(           )   \\ Coder! /\n |         |     -------\n |    |    |\n |    |    |\n(_____|_____)\n");
}
