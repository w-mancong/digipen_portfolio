/*!*****************************************************************************
\file bitset.hpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Assignment 2
\date 10-09-2022
\brief
This file contains function definition that mimics std::bitset functionalities
*******************************************************************************/
namespace HLP3
{
    /*!*****************************************************************************
        \brief Constructor for bitset
    *******************************************************************************/
    template <size_t N>
    bitset<N>::bitset()
    {
        bits = new uint8_t[pointer_size()]{};
    }

    /*!*****************************************************************************
        \brief Destructor for bitset
    *******************************************************************************/
    template <size_t N>
    bitset<N>::~bitset()
    {
        delete[] bits;
    }

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
    template <size_t N>
    bitset<N> &bitset<N>::set(size_t pos, bool value)
    {
        if(pos >= N)
            throw std::out_of_range("");
        // if N is 16, pos is 15, index will be 1
        size_t index = pos / 8;
        /*
            7 6 5 4 3 2 1 0  <- bit position
            - - - - - - - -
        */
        size_t bit_pos = pos % 8;
        *(bits + index) = value ? (*(bits + index) | (1 << bit_pos)) : (*(bits + index) & ~(1 << bit_pos));
        return *this;
    }

    /*!*****************************************************************************
        \brief Set the bit at position pos to false

        \param [in] pos: the position of the bit to set

        \return *this

        \exception std::out_of_range if pos does not correspond to a valid position
        within the bitset
    *******************************************************************************/
    template <size_t N>
    bitset<N> &bitset<N>::reset(size_t pos)
    {
        if(pos >= N)
            throw std::out_of_range("");
        size_t index = pos / 8, bit_pos = pos % 8;
        *(bits + index) &= ~(1 << bit_pos);
        return *this;
    }

    /*!*****************************************************************************
        \brief Flips bit, i.e. changes true value to false and vice versa. Flips
               the bit at the position pos

        \param [in] pos: the position of the bit to flip

        \return *this
        
        \exception std::out_of_range if pos does not correspond to a valid position
        within the bitset
    *******************************************************************************/
    template <size_t N>
    bitset<N> &bitset<N>::flip(size_t pos)
    {
        if(pos >= N)
            throw std::out_of_range("");
        size_t index = pos / 8, bit_pos = pos % 8;
        *(bits + index) ^= (1 << bit_pos);
        return *this;
    }

    /*!*****************************************************************************
        \brief Converts the contents of the bitset to a string. Uses zero to represent
               bits with value of false and one to represent bits with value of true.

        \param [in] zero: character to use to represent false
        \param [in] one:  character to use to represent true

        \return the converted string

        \exception May throw std::bad_alloc from the std::string constructor.
    *******************************************************************************/
    template <size_t N>
    std::string bitset<N>::to_string(char zero, char one) const
    {
        try {
            std::string str;
            for (int i = static_cast<int>(N - 1); i >= 0; --i)
            {
                size_t index = i / 8, bit_pos = i % 8;
                str += (*(bits + index) & (1 << bit_pos)) ? one : zero;
            }
            return str;
        } catch(std::bad_alloc){
            return "";
        }
    }

    /*!*****************************************************************************
        \brief Return the number of bits that the bitset holds

        \return number of bits that the bitset holds, i.e. the template parameter N.
    *******************************************************************************/
    template <size_t N>
    size_t bitset<N>::size() const
    {
        return N;
    }

    /*!*****************************************************************************
        \brief Returns the value of the bit at the position pos.Unlike operator[],
        performs a bounds check and throws std::out_of_range if pos does not correspond
        to a valid position in the bitset.

        \param [in] pos: position of the bit to return

        \return true if the requested bit is set, false otherwise.

        \exception std::out_of_range if pos does not correspond to a valid position
        within the bitset
    *******************************************************************************/
    template <size_t N>
    bool bitset<N>::test(size_t pos) const
    {
        if(pos >= N)
            throw std::out_of_range("");
        return this->operator[](pos);
    }

    /*!*****************************************************************************
        \brief Accesses the bit at position pos. Unlike test(), does not throw exceptions:
               the behavior is undefined if pos is out of bounds

        \param [in] pos: position of the bit to return

        \return the value of the requested bit
    *******************************************************************************/
    template <size_t N>
    bool bitset<N>::operator[](size_t pos) const
    {
        size_t index = pos / 8, bit_pos = pos % 8;
        return (*(bits + index) & (1 << bit_pos));
    }

    /*!*****************************************************************************
        \brief Returns the number of bits that are set to true

        \return Total number of bits that are set to true
    *******************************************************************************/
    template <size_t N>
    size_t bitset<N>::count() const
    {
        size_t counter { 0 };
        for (size_t i = 0; i < N; ++i)
        {
            size_t index = i / 8, bit_pos = i % 8;
            if( (*(bits + index) & (1 << bit_pos)) )
                ++counter;
        }
        return counter;
    }

    /*!*****************************************************************************
        \brief Calculate the size of pointer based on the value N
        If N == 8
        size of pointer will be 1
        
        If N == 35
        size of pointer will be 5

        \return size of pointer
    *******************************************************************************/
    template <size_t N>
    size_t bitset<N>::pointer_size()
    {
        size_t size = N / 8;
        // If N is a odd number and leaves a remainder, allocate another byte of memory
        if (N - (size * 8) > 0)
            ++size;
        return size;
    }
}