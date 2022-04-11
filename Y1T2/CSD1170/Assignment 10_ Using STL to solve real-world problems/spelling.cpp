/*!*****************************************************************************
\file spelling.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Assignment 10
\date 24-03-2022
\brief
This file contains functions definitions for spell_checker and string_utils
*******************************************************************************/
#include "spelling.hpp"

namespace hlp2
{
    /*!*****************************************************************************
    \brief
        Constructor that takes in a string as parameter
    \param [in] lexicon:
        Path to the file with all the relevant inputs
    *******************************************************************************/
    spell_checker::spell_checker(std::string const &lexicon) : dictionary{ lexicon } {}

    /*!*****************************************************************************
    \brief
        Count to total number of words starting with letter
    \param [in] letter:
        Words that are starting with the letter
    \param [in,out] count:
        The total number of words starting with letter will be stored here
    \return
        scrFILE_ERR_OPEN if file cannot be opened, else scrFILE_OK
    *******************************************************************************/
    spell_checker::SCResult spell_checker::words_starting_with(char letter, size_t &count) const
    {
        std::ifstream is{ dictionary };
        if(!is.is_open())
            return scrFILE_ERR_OPEN;

        count = 0;
        std::string buffer = {};
        letter = std::tolower(letter);
        while(std::getline(is, buffer))
        {
            char c = std::tolower(buffer[0]);
            if(c == letter)
                ++count;
        }

        is.close();
        return scrFILE_OK;
    }

    /*!*****************************************************************************
    \brief
        Accumulate the total number of words with lengths <= count and store it
        inside a vector
    \param [in,out] lengths:
        Incrementing the vector and using the word's length as index
    \param [in] count:
        Only increment words that are <= count
    \return
        scrFILE_ERR_OPEN if file cannot be opened, else scrFILE_OK
    *******************************************************************************/
    spell_checker::SCResult spell_checker::word_lengths(std::vector<size_t> &lengths, size_t count) const
    {
        std::ifstream is{dictionary};
        if (!is.is_open())
            return scrFILE_ERR_OPEN;

        std::string buffer = {};
        size_t index = {};
        while(std::getline(is, buffer))
        {
            if((index = buffer.size()) > count)
                continue;
            ++lengths[index];
        }

        is.close();
        return scrFILE_OK;
    }

    /*!*****************************************************************************
    \brief
        Retrieve the total number; shortest and longest words inside the input file
    \param [in,out] info:
        Storing information of total number of words, shortest and longest word
        of the input file
    \return
        scrFILE_ERR_OPEN if file cannot be opened, else scrFILE_OK
    *******************************************************************************/
    spell_checker::SCResult spell_checker::get_info(lexicon_info &info) const
    {
        std::ifstream is{dictionary};
        if (!is.is_open())
            return scrFILE_ERR_OPEN;

        std::string buffer = {};
        info.shortest = 100, info.longest = 0, info.count = 0;
        while(std::getline(is, buffer))
        {
            ++info.count;
            if(buffer.size() < info.shortest)
                info.shortest = buffer.size();
            if(buffer.size() > info.longest)
                info.longest = buffer.size();
        }
        
        is.close();
        return scrFILE_OK;
    }

    /*!*****************************************************************************
    \brief
        Check if the word is spelled correctly
    \param [in] word:
        Word to check if spelt correctly
    \return
        scrFILE_ERR_OPEN if file cannot be opened.
        If word cannot be found, scrWORD_BAD will be returned
        Else scrWORD_OK
    *******************************************************************************/
    spell_checker::SCResult spell_checker::spellcheck(std::string const &word) const
    {
        std::ifstream is{dictionary};
        if (!is.is_open())
            return scrFILE_ERR_OPEN;

        std::string buffer = {}, cmp = string_utils::upper_case(word);
        while(std::getline(is, buffer))
        {
            buffer = string_utils::upper_case(buffer);
            if(buffer > cmp)
                return scrWORD_BAD;
            else if (buffer == cmp)
                break;
        }

        is.close();
        return scrWORD_OK;
    }

    /*!*****************************************************************************
    \brief
        Check words with the same acronym in the precise order
    \param [in] acronym:
        Acronym of the words to find
    \param [in,out] words:
        Words containing the acronym will be stored inside this vector
    \param [in] maxlen:
        If size is 0, will check if any word length. Else will only check with words
        not longer than it
    \return
        scrFILE_ERR_OPEN if file cannot be opened, else scrFILE_OK
    *******************************************************************************/
    spell_checker::SCResult spell_checker::acronym_to_word(std::string const &acronym, std::vector<std::string> &words, size_t maxlen) const
    {
        std::ifstream is{dictionary};
        if (!is.is_open())
            return scrFILE_ERR_OPEN;

        std::string buffer = {}, cmp = string_utils::upper_case(acronym);
        while(std::getline(is, buffer))
        {
            if (maxlen && buffer.size() > maxlen)
                continue;
            std::string buffer2 = string_utils::upper_case(buffer);
            for (auto it1 = buffer2.begin(), it2 = cmp.begin(); it1 != buffer2.end() && buffer2[0] == cmp[0]; ++it1)
            {
                if(*it1 != *it2)
                    continue;
                ++it2;
                if(it2 < cmp.end())
                    continue;
                words.push_back(buffer);
                break;
            }
        }

        is.close();
        return scrFILE_OK;
    }

    /*!*****************************************************************************
    \brief
        Return a copy of str in upper cases
    \param [in] str:
        String to be turned to upper case
    \return
        A copy of str in upper case
    *******************************************************************************/
    std::string string_utils::upper_case(std::string const &str)
    {
        std::string tmp{ str };
        for (auto it = tmp.begin(); it != tmp.end(); ++it)
            *it = std::toupper(*it);
        return tmp;
    }

    /*!*****************************************************************************
    \brief
        For every white spaces, the word is split and store inside a vector
    \param [in] words:
        String containing a sentence to be split based on white spaces
    \return
        A vector of words after spliting the sentences into individual words
    *******************************************************************************/
    std::vector<std::string> string_utils::split(std::string const &words)
    {
        std::vector<std::string> res;
        std::string buffer = {};
        for (auto it = words.begin(); it != words.end(); ++it)
        {
            if(std::isspace(*it))
            {
                if(buffer.size())
                    res.push_back(buffer);
                buffer = {};
                continue;
            }
            buffer += *it;
        }
        if (buffer.size())
            res.push_back(buffer);
        return res;
    }
}