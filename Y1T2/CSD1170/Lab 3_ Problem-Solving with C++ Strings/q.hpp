/*!*****************************************************************************
\file q.cpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 2
\date 28-01-2022
\brief
This program make use of certain rules to turn english words into pig-latin words.
Functions included in the program are:
- IsVowel
Check if the letter is a vowel

- to_piglatin
Turn english words into pig-latin words
*******************************************************************************/
#ifndef	Q_HPP_
#define Q_HPP_

#include <string>

namespace hlp2
{
    /*!*****************************************************************************
	\brief
        Turns enlgish word into pig-latin word based on 5 special conditions
        1) If first letter is a vowel, append it with "-yay"
        2) If it starts with consonant, transfer all cosonant up to the first vowel
        and append it with "-ay"
        3) If it does not contain any vowel, append it with "-way"
        4) Special cases for Y. Treat it as a consonant if it's the first letter,
        else treat it as a vowel
        5) If the first letter of the word is upper case, the first letter of the
        pig-latin word should be upper case
	\param[in] word
        Current word to be changed into pig-latin word
    \return
        The pig-latin word
    *******************************************************************************/
    std::string to_piglatin(std::string word);
}

#endif