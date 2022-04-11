/*!*****************************************************************************
\file vector.hpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 5
\date 04-03-2022
\brief
This file contain function declarations for my own vector class
*******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
#ifndef VECTOR_HPP
#define VECTOR_HPP
////////////////////////////////////////////////////////////////////////////////
#include <cstddef>          // need this for size_t
#include <initializer_list> // need this for std::initializer_list<int>
 
namespace hlp2 
{
  class vector 
  {
    public:
      using value_type        = int;
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
}

#endif // VECTOR_HPP
