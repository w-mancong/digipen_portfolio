/*!*****************************************************************************
\file pa.hpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Assignment 4
\date 06-02-2022
\brief 
This program reads in five command-line parameters that will create and print
a generalized checkerboard [of characters] in a rectangular grid.
*******************************************************************************/
#ifndef PA_HPP_
#define PA_HPP_

#include <string>
#include <vector>

namespace hlp2
{
    using mystery_type = std::vector<std::vector<char>>;

  	/*!**************************************************************************
    \brief
		Create a rectanglar grid filled with starting characters to CYCLES
    
    \param [in] cmdline_params
      	Command lines input by the user
        cmdline_params[0]: rows
        cmdline_params[1]: cols
        cmdline_params[2]: starting character
        cmdline_params[3]: cycle
        cmdline_params[4]: width
    
    \return
		A vector storing a vector of chars that stores the information of
        the checkerboard
  	***************************************************************************/  
    mystery_type create_board(std::vector<std::string> const& cmdline_params);

  	/*!**************************************************************************
    \brief
		Prints out the checkerboard 

    \param [in] board
        Storing all the position of characters in the checkerboard

    \param [in] width
        Number of times to repeat printing the character width and row wise	
  	***************************************************************************/
    void print_board(mystery_type const& board, std::string const& width);
}

#endif