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
#include "q.hpp"

namespace Helper
{
    /*!*****************************************************************************
	\brief
        Check if the current character is a vowel. If it is a special case, include
        Y to be checked as well
	\param[in] c
        Character to be check if it's a vowel
    \param[in] special
        Based on special cases, Y would be considered a vowel
    \return
        True if the character c is a vowel
    *******************************************************************************/
    bool IsVowel(char const& c, bool special = false)
    {
        std::string vowels = "aeiou";
        if(special)
            vowels += 'y';
        char const ch = std::tolower(c);
        for(std::string::iterator it = vowels.begin(); it != vowels.end(); ++it)
        {
            if(*it == ch)
                return true;
        }
        return false;
    }
}

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
    std::string to_piglatin(std::string word)
    {
        std::string str(word);
        bool upper;
        if((upper = std::isupper(str[0])))
            str[0] = std::tolower(str[0]);
        if(Helper::IsVowel(str[0]))
            str += "-yay";
        else
        {
            bool gotVowel = false, first_is_y = str[0] == 'y' ? false : true;
            for(std::string::iterator it = str.begin(); it != str.end(); ++it)
            {
                if((gotVowel = Helper::IsVowel(*it, first_is_y)))
                    break;
            }
            if(gotVowel)
            {
                while(!Helper::IsVowel(str[0], first_is_y))
                {
                    const char c = str[0];
                    str.erase(0, 1);
                    str += c;
                }
                str += "-ay";
            }
            else
                str += "-way";
        }
        if(upper)
            str[0] = std::toupper(str[0]);
        return str;
    }
}