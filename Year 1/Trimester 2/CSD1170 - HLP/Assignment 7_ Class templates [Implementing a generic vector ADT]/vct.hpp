/*!*****************************************************************************
\file vct.hpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Assignment 7
\date 10-03-2022
\brief
This file contain function declarations for my own templated vector class
*******************************************************************************/
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
#ifndef VCT_HPP
#define VCT_HPP
////////////////////////////////////////////////////////////////////////////////
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <algorithm>

namespace hlp2 
{
  template <typename T>  
  class vector 
  {
    public:
        using value_type        = T;
        using size_type         = size_t;
        using reference         = value_type&;
        using const_reference   = const value_type&;
        using pointer           = value_type*;
        using const_pointer     = const value_type*;

    public:
        /*!*****************************************************************************
        \brief
            Default constructor of vector class
        *******************************************************************************/ 
        vector(void);

        /*!*****************************************************************************
        \brief
            Non-default constructor of vector class that takes in the size of data
            to be reserved

        \param n
            The total size to reserve for this vector
        *******************************************************************************/
        explicit vector(size_type n);

        /*!*****************************************************************************
        \brief
            Conversion constructor that converts an initializer_list to a vector class
        *******************************************************************************/
        vector(std::initializer_list<value_type> rhs);

        /*!*****************************************************************************
        \brief
            Copy constructor for vector class
        *******************************************************************************/
        vector(vector const& rhs);

        /*!*****************************************************************************
        \brief
            Destructor of vector class
        *******************************************************************************/
        ~vector(void);

        /*!*****************************************************************************
        \brief
            Copy assignment of vector class

        \return
            This vector class
        *******************************************************************************/ 
        vector&         operator=(vector const& rhs);

        /*!*****************************************************************************
        \brief
            Assignment of initializer_list for vector

        \return
            This vector class
        *******************************************************************************/
        vector&         operator=(std::initializer_list<value_type> rhs);

        /*!*****************************************************************************
        \brief
            Accessor and mutator method
        
        \param index
            Index of the data inside the pointer
    
        \return
            A reference to the data at position index
        *******************************************************************************/ 
        reference       operator[](size_type index);

        /*!*****************************************************************************
        \brief
            Accessor method

        \param index
            Index of the data inside the pointer

        \return
            A constant reference to the data at position index
        *******************************************************************************/ 
        const_reference operator[](size_type index) const;
        
        /*!*****************************************************************************
        \brief
            Add an element to the back of the vector

        \param value
            Element to be added
        *******************************************************************************/
        void            push_back(value_type value);

        /*!*****************************************************************************
        \brief
            Remove one element from the back
        *******************************************************************************/
        void            pop_back(void);

        /*!*****************************************************************************
        \brief
            Swap the data of this class with rhs

        \param rhs
            Data to swap with
        *******************************************************************************/
        void            swap(vector& rhs);

        /*!*****************************************************************************
        \brief
            Reserve a total space inside the vector

        \param newsize
            The newsize for vectors to reserve space for
        *******************************************************************************/ 
        void            reserve(size_type newsize);

        /*!*****************************************************************************
        \brief
            Resize the vector so that it contains newsize elements

        \param newsize
            The new total elements inside the vector
        *******************************************************************************/
        void            resize(size_type newsize);
        
        /*!*****************************************************************************
        \brief
            Check if the vector is empty

        \return 
            True if the vector is empty
        *******************************************************************************/ 
        bool            empty(void) const;

        /*!*****************************************************************************
        \brief
            Returns the total number of elements inside the vector

        \return
            Total elements inside the vector
        *******************************************************************************/
        size_type       size(void) const;

        /*!*****************************************************************************
        \brief
            Returns the total space available inside the vector

        \return
            Total space available inside the vector
        *******************************************************************************/ 
        size_type       capacity(void) const;

        /*!*****************************************************************************
        \brief
            Return the total number of allocations done for this vector

        \return
            Total number of allocations done
        *******************************************************************************/ 
        size_type       allocations(void) const;
        
        /*!*****************************************************************************
        \brief
            Returns a pointer to the first element of the vector

        \return
            Pointer to the first element of the vector
        *******************************************************************************/
        pointer         begin(void);

        /*!*****************************************************************************
        \brief
            Returns a pointer to the one past the last element of the vector

        \return
            Pointer to the one past the last element of the vector
        *******************************************************************************/ 
        pointer         end(void);

        /*!*****************************************************************************
        \brief
            Returns a const pointer to the first element of the vector

        \return
            A constant pointer to the first element of the vector
        *******************************************************************************/ 
        const_pointer   begin(void) const;

        /*!*****************************************************************************
        \brief
            Returns a constant pointer to the one past the last element of the vector

        \return
            A constant pointer to the one past the last element of the vector
        *******************************************************************************/ 
        const_pointer   end(void) const;

        /*!*****************************************************************************
        \brief
            Returns a const pointer to the first element of the vector

        \return
            A constant pointer to the first element of the vector
        *******************************************************************************/ 
        const_pointer   cbegin(void) const;

        /*!*****************************************************************************
        \brief
            Returns a constant pointer to the one past the last element of the vector
        
        \return
            A constant pointer to the one past the last element of the vector
        *******************************************************************************/
        const_pointer   cend(void) const;

    private:
        /*!*****************************************************************************
        \brief
            Helper function to get the data at position index
    
        \param index
            Index of the data inside the pointer
    
        \return
            A constant pointer to the data at position index
        *******************************************************************************/ 
        const_pointer   get(size_type index) const;

        /*!*****************************************************************************
        \brief
            Helper function to copy data from another vector to this vector

        \param rhs
            Reference to the other vector
        *******************************************************************************/ 
        void            copy_data(vector const& rhs);

        /*!*****************************************************************************
        \brief
            Helper function to copy data from initializer_list

        \param rhs
            Reference to the initializer_list
        *******************************************************************************/ 
        void            copy_data(std::initializer_list<value_type> const& rhs);
        
        size_type space;  // the allocated size (in terms of elements) of the array
        size_type allocs; // number of times space has been updated
        size_type sz;     // the number of elements in the array
        pointer   data;   // the dynamically allocated array
    };

    /*!*****************************************************************************
    \brief
        Default constructor of vector class
    *******************************************************************************/
    template <typename T>
    vector<T>::vector(void) : space{ 0 }, allocs{ 0 }, sz{ space }, data{ nullptr }
    {
        // std::cout << "default constructor" << std::endl;
    }

    /*!*****************************************************************************
    \brief
        Non-default constructor of vector class that takes in the size of data
        to be reserved

    \param n
        The total size to reserve for this vector
    *******************************************************************************/ 
    template <typename T>
    vector<T>::vector(size_type n) : space{ 0 }, allocs{ 0 }, sz{ n }, data{ nullptr }
    {
        reserve(n);
    }

    /*!*****************************************************************************
    \brief
        Conversion constructor that converts an initializer_list to a vector class
    *******************************************************************************/ 
    template <typename T>
    vector<T>::vector(std::initializer_list<value_type> rhs) : space{ 0 }, allocs{ 0 }, sz{ rhs.size() }, data{ nullptr }
    {
        reserve(rhs.size());
        copy_data(rhs);
    }

    /*!*****************************************************************************
    \brief
        Copy constructor for vector class
    *******************************************************************************/ 
    template <typename T>
    vector<T>::vector(vector const& rhs) : space{ 0 }, allocs { 0 }, sz{ rhs.sz }, data{ nullptr }
    {
        // std::cout << "copy constructor\n";
        reserve(rhs.sz);
        copy_data(rhs);
    }

    /*!*****************************************************************************
    \brief
        Destructor of vector class
    *******************************************************************************/ 
    template <typename T>
    vector<T>::~vector(void)
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
    template <typename T>
    vector<T>& vector<T>::operator=(vector const& rhs)
    {
        vector<T> temp { rhs };
        temp.space = rhs.sz, ++(temp.allocs = allocs);
        swap(temp);
        return *this;
    }

    /*!*****************************************************************************
    \brief
        Assignment of initializer_list for vector

    \return
        This vector class
    *******************************************************************************/
    template <typename T>
    vector<T>& vector<T>::operator=(std::initializer_list<value_type> rhs)
    {
        vector<T> temp { rhs };
        temp.sz = rhs.size(), temp.space = sz;
        swap(temp);
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
    template <typename T>
    typename vector<T>::reference vector<T>::operator[](size_type index)
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
    template <typename T>
    typename vector<T>::const_reference vector<T>::operator[](size_type index) const
    {
        return *get(index);
    }

    /*!*****************************************************************************
    \brief
        Add an element to the back of the vector

    \param value
        Element to be added
    *******************************************************************************/ 
    template <typename T>
    void vector<T>::push_back(value_type value)
    {
        if(sz == space && space) reserve(space << 1);
        else                     reserve(1);   // space is 0
        *(data + sz++) = value;
    }

    /*!*****************************************************************************
    \brief
        Remove one element from the back
    *******************************************************************************/
    template <typename T>
    void vector<T>::pop_back(void)
    {
        if(empty())
            return;
        --sz;
    }

    /*!*****************************************************************************
    \brief
        Swap the data of this class with rhs

    \param rhs
        Data to swap with
    *******************************************************************************/
    template <typename T>
    void vector<T>::swap(vector& rhs)
    {
        std::swap(space, rhs.space);
        std::swap(allocs, rhs.allocs);
        std::swap(sz, rhs.sz);
        std::swap(data, rhs.data);
    }

    /*!*****************************************************************************
    \brief
        Reserve a total space inside the vector

    \param newsize
        The newsize for vectors to reserve space for
    *******************************************************************************/ 
    template <typename T>
    void vector<T>::reserve(size_type newsize)
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
    template <typename T>
    void vector<T>::resize(size_type newsize)
    {
        if(newsize > space) reserve(newsize);
        else if(newsize > sz && newsize <= space)
        {
            for(size_type i = 0, index = sz; i < newsize - sz; ++i, ++index)
                *(data + index) = {};
        }
        sz = newsize;
    }

    /*!*****************************************************************************
    \brief
        Check if the vector is empty

    \return 
        True if the vector is empty
    *******************************************************************************/ 
    template <typename T>
    bool vector<T>::empty(void) const
    {
        return !sz;
    }

    /*!*****************************************************************************
    \brief
        Returns the total number of elements inside the vector

    \return
        Total elements inside the vector
    *******************************************************************************/ 
    template <typename T>
    typename vector<T>::size_type vector<T>::size(void) const
    {
        return sz;
    }

    /*!*****************************************************************************
    \brief
        Returns the total space available inside the vector

    \return
        Total space available inside the vector
    *******************************************************************************/ 
    template <typename T>
    typename vector<T>::size_type vector<T>::capacity(void) const
    {
        return space;
    }

    /*!*****************************************************************************
    \brief
        Return the total number of allocations done for this vector

    \return
        Total number of allocations done
    *******************************************************************************/ 
    template <typename T>
    typename vector<T>::size_type vector<T>::allocations(void) const
    {
        return allocs;
    }

    /*!*****************************************************************************
    \brief
        Returns a pointer to the first element of the vector

    \return
        Pointer to the first element of the vector
    *******************************************************************************/ 
    template <typename T>
    typename vector<T>::pointer vector<T>::begin(void)
    {
        return const_cast<pointer>(cbegin());
    }

    /*!*****************************************************************************
    \brief
        Returns a pointer to the one past the last element of the vector

    \return
        Pointer to the one past the last element of the vector
    *******************************************************************************/ 
    template <typename T>
    typename vector<T>::pointer vector<T>::end(void)
    {
        return const_cast<pointer>(cend());
    }

    /*!*****************************************************************************
    \brief
        Returns a const pointer to the first element of the vector

    \return
        A constant pointer to the first element of the vector
    *******************************************************************************/ 
    template <typename T>
    typename vector<T>::const_pointer vector<T>::begin(void) const
    {
        return cbegin();
    }

    /*!*****************************************************************************
    \brief
        Returns a constant pointer to the one past the last element of the vector

    \return
        A constant pointer to the one past the last element of the vector
    *******************************************************************************/ 
    template <typename T>
    typename vector<T>::const_pointer vector<T>::end(void) const
    {
        return cend();
    }

    /*!*****************************************************************************
    \brief
        Returns a const pointer to the first element of the vector

    \return
        A constant pointer to the first element of the vector
    *******************************************************************************/ 
    template <typename T>
    typename vector<T>::const_pointer vector<T>::cbegin(void) const
    {
        return data;
    }

    /*!*****************************************************************************
    \brief
        Returns a constant pointer to the one past the last element of the vector

    \return
        A constant pointer to the one past the last element of the vector
    *******************************************************************************/ 
    template <typename T>
    typename vector<T>::const_pointer vector<T>::cend(void) const
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
    template <typename T>
    typename vector<T>::const_pointer vector<T>::get(size_type index) const
    {
        return data + index;
    }

    /*!*****************************************************************************
    \brief
        Helper function to copy data from another vector to this vector

    \param rhs
        Reference to the other vector
    *******************************************************************************/ 
    template <typename T>
    void vector<T>::copy_data(vector const& rhs)
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
    template <typename T>
    void vector<T>::copy_data(std::initializer_list<value_type> const& rhs)
    {
        pointer ptr = data;
        for (value_type const& vt : rhs)
        {
            *ptr++ = vt;
        }
    }
} // namespace hlp2

#endif // VCT_HPP
