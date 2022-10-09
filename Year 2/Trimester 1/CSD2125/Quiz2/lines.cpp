/*!*****************************************************************************
\file lines.hpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Quiz 1
\date 02-09-2022
\brief
This file contain functions that count the number of lines from the text file
*******************************************************************************/
#include "lines.hpp"

namespace HLP3
{
    /*!*****************************************************************************
        \brief
        Calculate the total number of lines in the file

        \param [in] fileName: array of pointers to C-strings

        \return Total number of lines in the files
    *******************************************************************************/
    int lines(char const **fileName)
    {
        int counter{0};
        std::string buffer;
        while (*fileName)
        {
            std::ifstream ifs{ *(fileName++) };
            while(std::getline(ifs, buffer))
                ++counter;
        }
        return counter;
    }
}