/*!*****************************************************************************
\file q.hpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Assignment 1
\date 15-01-2022
\brief
This program reads a text file specified by it's parameter input_filename
and prints some statistical results about it's contents to an output file
specified by parameter analysis_file. The functions include:
- seperators
prints out a set amount of separators between each row of data
- q
Opens a file and output statistical results of it's content
*******************************************************************************/
#ifndef Q_HPP_
#define Q_HPP_

namespace hlp2
{
    /*!*****************************************************************************
    \brief
    Opens a file and output it's contents as a statistical result
    \param[in] input_filename
    File to be opened
    \param[in] analysis_file
    File that will have the statistical result written into 
    *******************************************************************************/
    void q(char const* input_filename, char const* analysis_file);
}

#endif