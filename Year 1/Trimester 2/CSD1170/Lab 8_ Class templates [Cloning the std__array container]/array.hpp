/*!*****************************************************************************
\file array.hpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 8
\date 13-03-2022
\brief
    This file defines a templated class array that mimics a standard C style
    static array with additional function features
*******************************************************************************/

//-------------------------------------------------------------------------
#ifndef ARRAY_HPP
#define ARRAY_HPP
//-------------------------------------------------------------------------
#include <cstddef> // for size_t
#include <initializer_list>

namespace hlp2
{
    template <typename T, size_t N>
    class Array
    {
    private:
        using const_class_reference = Array<T, N> const&;
    public:
        using value_type            = T;
        using reference             = value_type&;
        using const_reference       = const value_type&;
        using pointer               = value_type*;
        using const_pointer         = const value_type*;
        using iterator              = pointer;
        using const_iterator        = const_pointer;
        using size_type             = size_t;

    public:
        /*!*****************************************************************************
        \brief
            Default constructor of array class
        *******************************************************************************/ 
        Array(void);

        /*!*****************************************************************************
        \brief
            Conversion constructor that converts an initializer_list to an array class       
        \param [in] rhs
            Initializer list that takes in value_type
        *******************************************************************************/ 
        Array(std::initializer_list<value_type> const &rhs);

        /*!*****************************************************************************
        \brief
            Destructor
        *******************************************************************************/
        ~Array(void);

        /*!*****************************************************************************
        \brief
            Copy constructor
        \param [in] rhs
            Array object to copy it's content over
        *******************************************************************************/
        Array(Array const& rhs);

        /*!*****************************************************************************
        \brief
            Overloaded copy assignment operator
        \param [in] rhs
            Array object to copy it's content over
        *******************************************************************************/
        Array& operator=(Array const& rhs);

        /*!*****************************************************************************
        \brief
            Get a pointer to the first element of the array
        \return
            The address of the first element
        *******************************************************************************/
        iterator        begin(void);

        /*!*****************************************************************************
        \brief
            Get a const pointer to the first element of the array
        \return
            The address of the first element
        *******************************************************************************/
        const_iterator  begin(void) const;

        /*!*****************************************************************************
        \brief
            Get a pointer to one past the last element of the array
        \return
            The address to the end of the array
        *******************************************************************************/        
        iterator        end(void);

        /*!*****************************************************************************
        \brief
            Get a const pointer to one past the last element of the array
        \return
            The address to the end of the array
        *******************************************************************************/
        const_iterator  end(void) const;

        /*!*****************************************************************************
        \brief
            Get a const pointer to the first element of the array
        \return
            The address of the first element
        *******************************************************************************/
        const_iterator  cbegin(void) const;

        /*!*****************************************************************************
        \brief
            Get a const pointer to one past the last element of the array
        \return
            The address to the end of the array
        *******************************************************************************/
        const_iterator  cend(void) const;

        /*!*****************************************************************************
        \brief
            A reference to the first element of the array
        \return
            A reference to value_type
        *******************************************************************************/
        reference       front(void);

        /*!*****************************************************************************
        \brief
            A const reference to the first element of the array
        \return
            A reference to value_type
        *******************************************************************************/
        const_reference front(void) const;

        /*!*****************************************************************************
        \brief
            A reference to the last element of the array
        \return
            A reference to value_type
        *******************************************************************************/
        reference       back(void);

        /*!*****************************************************************************
        \brief
            A const reference to the last element of the array
        \return
            A reference to value_type
        *******************************************************************************/
        const_reference back(void) const;

        /*!*****************************************************************************
        \brief
            A reference to the element at the specified index
        \param [in] index
            Position of the element in the array
        \return
            A reference to value_type
        *******************************************************************************/
        reference       operator[](size_type index);

        /*!*****************************************************************************
        \brief
            A const reference to the element at the specified index
        \param [in] index
            Position of the element in the array
        \return
            A reference to value_type
        *******************************************************************************/
        const_reference operator[](size_type index) const;

        /*!*****************************************************************************
        \brief
            Check if the array is empty
        \return
            True if the array is empty, else false
        *******************************************************************************/
        bool            empty(void) const;

        /*!*****************************************************************************
        \brief
            Total number of element containing in the array
        \return
            Number of elements in the array
        *******************************************************************************/
        size_type       size(void) const;

        /*!*****************************************************************************
        \brief
            Fill the entire array with val
        \param [in] val
            Data to fill the entire array with
        *******************************************************************************/
        void            fill(value_type const& val);

        /*!*****************************************************************************
        \brief
            Swap the data in this class with rhs
        \param [in] rhs
            The class to have it's data swap with
        *******************************************************************************/
        void            swap(Array& rhs);

    private:
        value_type      data[N];
    };

#include "array.tpp"
} // end namespace hlp2

#endif
