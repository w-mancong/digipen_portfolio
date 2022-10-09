/*!*****************************************************************************
\file bitset.h
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Assignment 2
\date 10-09-2022
\brief
This file contains function declarations that mimics std::bitset functionalities
*******************************************************************************/
#ifndef BITSET_H
#define BITSET_H

namespace HLP3 
{
    template <size_t N>
    class bitset
    {
    public:
        /*!*****************************************************************************
            \brief Constructor for bitset
        *******************************************************************************/
        bitset();

        /*!*****************************************************************************
            \brief Destructor for bitset
        *******************************************************************************/
        ~bitset();

        /*
                                  7 6 5 4 3 2 1 0  <-  bit position
            Most Significant -->  - - - - - - - -  <-- Least Significant
        */
        /*!*****************************************************************************
            \brief Sets the bit at position pos to the value value.

            \param [in] pos: the position of the bit to set (least significant to most significant)
            \param [in] value: the value to set the bit to

            \return *this

            \exception std::out_of_range if pos does not correspond to a valid position
            within the bitset
        *******************************************************************************/
        bitset &set(size_t pos, bool value = true);

        /*!*****************************************************************************
            \brief Set the bit at position pos to false

            \param [in] pos: the position of the bit to set

            \return *this

            \exception std::out_of_range if pos does not correspond to a valid position
            within the bitset
        *******************************************************************************/
        bitset &reset(size_t pos);

        /*!*****************************************************************************
            \brief Flips bit, i.e. changes true value to false and vice versa. Flips
                   the bit at the position pos

            \param [in] pos: the position of the bit to flip

            \return *this

            \exception std::out_of_range if pos does not correspond to a valid position
            within the bitset
        *******************************************************************************/
        bitset &flip(size_t pos);

        /*!*****************************************************************************
            \brief Converts the contents of the bitset to a string. Uses zero to represent
                   bits with value of false and one to represent bits with value of true.

            \param [in] zero: character to use to represent false
            \param [in] one:  character to use to represent true

            \return the converted string

            \exception May throw std::bad_alloc from the std::string constructor.
        *******************************************************************************/
        std::string to_string(char zero = '0', char one = '1') const;

        /*!*****************************************************************************
            \brief Return the number of bits that the bitset holds

            \return number of bits that the bitset holds, i.e. the template parameter N.
        *******************************************************************************/
        size_t size() const;

        /*!*****************************************************************************
            \brief Returns the value of the bit at the position pos.Unlike operator[],
            performs a bounds check and throws std::out_of_range if pos does not correspond
            to a valid position in the bitset.

            \param [in] pos: position of the bit to return

            \return true if the requested bit is set, false otherwise.

            \exception std::out_of_range if pos does not correspond to a valid position
            within the bitset
        *******************************************************************************/
        bool test(size_t pos) const;

        /*!*****************************************************************************
            \brief Accesses the bit at position pos. Unlike test(), does not throw exceptions:
                   the behavior is undefined if pos is out of bounds

            \param [in] pos: position of the bit to return

            \return the value of the requested bit
        *******************************************************************************/
        bool operator[](size_t pos) const;

        /*!*****************************************************************************
            \brief Returns the number of bits that are set to true

            \return Total number of bits that are set to true
        *******************************************************************************/
        size_t count() const;

    private:
        /*!*****************************************************************************
            \brief Calculate the size of pointer based on the value N
            If N == 8
            size of pointer will be 1

            If N == 35
            size of pointer will be 5

            \return size of pointer
        *******************************************************************************/
        size_t pointer_size();

        uint8_t *bits;
    };
}

#include "bitset.hpp"

#endif

