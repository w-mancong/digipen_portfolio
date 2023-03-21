/*!*****************************************************************************
\file q.hpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 1
\date 14-01-2022
\brief
This program takes in a message and decode a secret message 
hidden steganographically. The functions include:
- q
Opens and read text from file. Based on the hard coded keywords, the function
will print out the following words in sequence
*******************************************************************************/
#ifndef Q_HPP_
#define Q_HPP_

namespace hlp2 
{
	/*!*****************************************************************************
    \brief
    Reads text and print out decoded message
    \param[in] filename
    name of the file to be opened
    \param[in] key_words
    special key_words that reveals the hidden message
    *******************************************************************************/
	void q(char const *filename, const char **key_words);
}

#endif
