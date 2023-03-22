/*!*****************************************************************************
\file spelling.hpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Assignment 10
\date 24-03-2022
\brief
This file contains class and functions declaration for spell_checker
and string_utils
*******************************************************************************/
#ifndef SPELLING_HPP
#define SPELLING_HPP

#include <string>
#include <vector>
#include <fstream>

namespace hlp2
{
    class spell_checker
    {
    public:
        enum SCResult
        {
            scrFILE_OK          = -1,
            scrFILE_ERR_OPEN    = -2,
            scrWORD_OK          = 1,
            scrWORD_BAD         = 2,
        };

        struct lexicon_info
        {
            size_t shortest;    // Shortest word in lexicon
            size_t longest;     // Longest word in lexicon
            size_t count;       // Number of words in lexicon
        };

        /*!*****************************************************************************
        \brief
            Constructor that takes in a string as parameter
        \param [in] lexicon:
            Path to the file with all the relevant inputs
        *******************************************************************************/
        spell_checker(std::string const &lexicon);

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
        SCResult words_starting_with(char letter, size_t &count) const;

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
        SCResult word_lengths(std::vector<size_t> &lengths, size_t count) const;

        /*!*****************************************************************************
        \brief
            Retrieve the total number; shortest and longest words inside the input file
        \param [in,out] info:
            Storing information of total number of words, shortest and longest word
            of the input file
        \return 
            scrFILE_ERR_OPEN if file cannot be opened, else scrFILE_OK
        *******************************************************************************/
        SCResult get_info(lexicon_info &info) const;

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
        SCResult spellcheck(std::string const &word) const;

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
        SCResult acronym_to_word(std::string const &acronym, std::vector<std::string> &words, size_t maxlen = 0) const;
    private:
        std::string dictionary;
    };

    class string_utils
    {
    public:
        /*!*****************************************************************************
        \brief
            Return a copy of str in upper cases
        \param [in] str:
            String to be turned to upper case
        \return
            A copy of str in upper case
        *******************************************************************************/
        static std::string upper_case(std::string const &str);

        /*!*****************************************************************************
        \brief
            For every white spaces, the word is split and store inside a vector
        \param [in] words:
            String containing a sentence to be split based on white spaces
        \return
            A vector of words after spliting the sentences into individual words
        *******************************************************************************/
        static std::vector<std::string> split(std::string const &words);
    };
}

#endif