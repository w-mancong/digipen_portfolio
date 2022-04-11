/*!*****************************************************************************
\file vector.hpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 5
\date 04-03-2022
\brief
This file contain function definitions for my own vector class
*******************************************************************************/
#include "vector.hpp"
#include <iostream>

namespace hlp2
{
    /*!*****************************************************************************
    \brief
        Default constructor of vector class
    *******************************************************************************/ 
    vector::vector(void) : space{ 0 }, allocs{ 0 }, sz{ space }, data{ nullptr }
    {
        
    }

    /*!*****************************************************************************
    \brief
        Non-default constructor of vector class that takes in the size of data
        to be reserved
    
    \param n
        The total size to reserve for this vector
    *******************************************************************************/ 
    vector::vector(size_type n) : space{ 0 }, allocs{ 0 }, sz{ n }, data{ nullptr }
    {
        reserve(n);
    }

    /*!*****************************************************************************
    \brief
        Conversion constructor that converts an initializer_list to a vector class
    *******************************************************************************/ 
    vector::vector(std::initializer_list<value_type> rhs) : space{ 0 }, allocs{ 0 }, sz{ rhs.size() }, data{ nullptr }
    {
        reserve(rhs.size());
        copy_data(rhs);
    }

    /*!*****************************************************************************
    \brief
        Copy constructor for vector class
    *******************************************************************************/ 
    vector::vector(vector const& rhs) : space{ 0 }, allocs { 0 }, sz{ rhs.sz }, data{ nullptr }
    {
        reserve(rhs.sz);
        copy_data(rhs);
    }

    /*!*****************************************************************************
    \brief
        Destructor of vector class
    *******************************************************************************/ 
    vector::~vector(void)
    {
        if(data)
        {
            delete[] data;
            data = nullptr;
        }
    }

    /*!*****************************************************************************
    \brief
        Copy assignment of vector class

    \return
        This vector class
    *******************************************************************************/ 
    vector& vector::operator=(vector const& rhs)
    {
        sz = rhs.sz, space = sz, ++allocs;
        if (rhs.sz > space)
            reserve(rhs.sz);
        if (rhs.data != data)
            copy_data(rhs);

        return *this;
    }

    /*!*****************************************************************************
    \brief
        Assignment of initializer_list for vector
    
    \return
        This vector class
    *******************************************************************************/ 
    vector& vector::operator=(std::initializer_list<value_type> rhs)
    {
        reserve(rhs.size()), sz = rhs.size(), space = sz;
        copy_data(rhs);
        return *this;
    }

    /*!*****************************************************************************
    \brief
        Accessor and mutator method
    
    \param index
        Index of the data inside the pointer

    \return
        A reference to the data at position index
    *******************************************************************************/ 
    vector::reference vector::operator[](size_type index)
    {
        return *const_cast<pointer>(get(index));
    }

    /*!*****************************************************************************
    \brief
        Accessor method

    \param index
        Index of the data inside the pointer

    \return
        A constant reference to the data at position index
    *******************************************************************************/ 
    vector::const_reference vector::operator[](size_type index) const
    {
        return *get(index);
    }

    /*!*****************************************************************************
    \brief
        Add an element to the back of the vector

    \param value
        Element to be added
    *******************************************************************************/ 
    void vector::push_back(value_type value)
    {
        if(sz == space && space) reserve(space << 1);
        else                     reserve(1);   // space is 0
        *(data + sz++) = value;
    }

    /*!*****************************************************************************
    \brief
        Reserve a total space inside the vector

    \param newsize
        The newsize for vectors to reserve space for
    *******************************************************************************/ 
    void vector::reserve(size_type newsize)
    {
        if(newsize <= space)
            return; 
        pointer temp = new value_type[newsize];
        for(size_type i = 0; data && i < sz; ++i)
            *(temp + i) = *(data + i);
        if(data)
            delete[] data;
        data = temp;
        space = newsize, ++allocs;
    }

    /*!*****************************************************************************
    \brief
        Resize the vector so that it contains newsize elements

    \param newsize
        The new total elements inside the vector
    *******************************************************************************/ 
    void vector::resize(size_type newsize)
    {
        if(newsize > space) reserve(newsize);
        else if(newsize > sz && newsize <= space)
        {
            for(size_type i = 0, index = sz; i < newsize - sz; ++i, ++index)
                *(data + index) = 0;
            sz = newsize;
        }
        else if(newsize < sz)
            sz = newsize;
    }

    /*!*****************************************************************************
    \brief
        Check if the vector is empty

    \return 
        True if the vector is empty
    *******************************************************************************/ 
    bool vector::empty(void) const
    {
        return !sz;
    }

    /*!*****************************************************************************
    \brief
        Returns the total number of elements inside the vector

    \return
        Total elements inside the vector
    *******************************************************************************/ 
    vector::size_type vector::size(void) const
    {
        return sz;
    }

    /*!*****************************************************************************
    \brief
        Returns the total space available inside the vector

    \return
        Total space available inside the vector
    *******************************************************************************/ 
    vector::size_type vector::capacity(void) const
    {
        return space;
    }

    /*!*****************************************************************************
    \brief
        Return the total number of allocations done for this vector
    
    \return
        Total number of allocations done
    *******************************************************************************/ 
    vector::size_type vector::allocations(void) const
    {
        return allocs;
    }

    /*!*****************************************************************************
    \brief
        Returns a pointer to the first element of the vector

    \return
        Pointer to the first element of the vector
    *******************************************************************************/ 
    vector::pointer vector::begin(void)
    {
        return const_cast<pointer>(cbegin());
    }

    /*!*****************************************************************************
    \brief
        Returns a pointer to the one past the last element of the vector
    
    \return
        Pointer to the one past the last element of the vector
    *******************************************************************************/ 
    vector::pointer vector::end(void)
    {
        return const_cast<pointer>(cend());
    }

    /*!*****************************************************************************
    \brief
        Returns a const pointer to the first element of the vector
    
    \return
        A constant pointer to the first element of the vector
    *******************************************************************************/ 
    vector::const_pointer vector::begin(void) const
    {
        return cbegin();
    }

    /*!*****************************************************************************
    \brief
        Returns a constant pointer to the one past the last element of the vector
    
    \return
        A constant pointer to the one past the last element of the vector
    *******************************************************************************/ 
    vector::const_pointer vector::end(void) const
    {
        return cend();
    }

    /*!*****************************************************************************
    \brief
        Returns a const pointer to the first element of the vector
    
    \return
        A constant pointer to the first element of the vector
    *******************************************************************************/ 
    vector::const_pointer vector::cbegin(void) const
    {
        return data;
    }


    vector::const_pointer vector::cend(void) const
    {
        return data + sz;
    }

    /*!*****************************************************************************
    \brief
        Helper function to get the data at position index

    \param index
        Index of the data inside the pointer

    \return
        A constant pointer to the data at position index
    *******************************************************************************/ 
    vector::const_pointer vector::get(size_type index) const
    {
        return data + index;
    }

    /*!*****************************************************************************
    \brief
        Helper function to copy data from another vector to this vector
    
    \param rhs
        Reference to the other vector
    *******************************************************************************/ 
    void vector::copy_data(vector const& rhs)
    {
        pointer ptr = data;
        for (size_type i = 0; i < rhs.sz; ++i)
            *(ptr + i) = *(rhs.data + i);
    }

    /*!*****************************************************************************
    \brief
        Helper function to copy data from initializer_list

    \param rhs
        Reference to the initializer_list
    *******************************************************************************/ 
    void vector::copy_data(std::initializer_list<value_type> const& rhs)
    {
        pointer ptr = data;
        for (value_type vt : rhs)
            *(ptr++) = vt;
    }
}