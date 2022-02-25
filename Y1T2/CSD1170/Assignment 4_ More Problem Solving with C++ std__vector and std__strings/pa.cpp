/*!*****************************************************************************
\file pa.cpp
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
#include <iostream>
#include "pa.hpp"

namespace hlp2
{    
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
    mystery_type create_board(std::vector<std::string> const& cmdline_params)
    {
        if(5 > cmdline_params.size())
        {
            std::cerr << "Usage: ./program-name rows cols start cycle width" << std::endl;
            return mystery_type();
        }
        
        const int ROW = std::stoi(cmdline_params[0]), COL = std::stoi(cmdline_params[1]), CYCLE = std::stoi(cmdline_params[3]), WIDTH = std::stoi(cmdline_params[4]);
        const char CHAR = cmdline_params[2][0];
        
        if(0 >= ROW || 0 >= COL || 0 >= CYCLE || 0 >= WIDTH || 'z' < (int)CHAR + CYCLE)
            return mystery_type();
        
        mystery_type board;
        
        for (int i = 0; i < ROW; ++i)
        {
            std::vector<char> str;
            for (int j = 0; j < COL; ++j)
            {
                char c = CHAR + (i + j) % CYCLE;
                str.push_back(c);
            }
            board.push_back(str);
        }
        return board;
    }

  	/*!**************************************************************************
    \brief
		Prints out the checkerboard 

    \param [in] board
        Storing all the position of characters in the checkerboard

    \param [in] width
        Number of times to repeat printing the character width and row wise	
  	***************************************************************************/ 
    void print_board(mystery_type const& board, std::string const& width)
    {
        size_t WIDTH;
        try
        {
            WIDTH = std::stoll(width);
        }
        catch(std::invalid_argument& e)
        {
            return;
        }
        for(size_t i = 0; i < board.size(); ++i)
        {
            for(size_t j = 0; j < WIDTH; ++j)
            {
                for(size_t k = 0; k < board[i].size(); ++k)
                {
                    for(size_t l = 0; l < WIDTH; ++l)
                    {
                        std::cout << board[i][k];
                    }
                }
                std::cout << std::endl;
            }
        }
    }
}