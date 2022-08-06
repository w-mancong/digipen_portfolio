/*!*****************************************************************************
\file ft.hpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 5
\date 05-03-2022
\brief
This file contains different template implementations to:
  copy 
    - data from half-open range array into a destination
  count
    - total number of data inside the half-open range array
  display
    - the contents inside the half-open range array
  equal
    - check if the two arrays contains the same data
  find
    - the first instance of the value inside the half-open range array
  fill
    - the entire half-open range array with value
  max_element
    - search for the biggest element inside the half-open range array
  min_element
    - search for the smallest element inside the half-open range array
  remove
    - all instances of values inside the half-open range array
  replace
    - all instances of old value with new value
  sum
    - up the total value inside the half-open range array
  swap
    - the data of two values
  swap_ranges
    - swap the data of two arrays
*******************************************************************************/

//-------------------------------------------------------------------------
#ifndef FT_H
#define FT_H
//-------------------------------------------------------------------------
#include <iostream>

namespace hlp2 
{
  /*!*****************************************************************************
  \brief
    Copy the data from the half-open range array into the dst

  \param beg
    Pointer to the start of the half-open range array
  \param end
    Pointer to the end of the half-open range array
  \param dst
    Pointer the the start of the array to have the data be copied over to

  \return 
    A pointer to the end of the dst
  *******************************************************************************/ 
  template <typename T> T* copy(T const* beg, T const* end, T* dst);
  
  /*!*****************************************************************************
  \brief
    Iterate through the half-open range array and return the total number
    of val in it
    
  \param beg
    Pointer to the start of the half-open range array
  \param end
    Pointer to the end of the half-open range array
  \param val
    The value to be counted inside the half-open range array

  \return
    The total number of val inside the half-open range array
  *******************************************************************************/ 
  template <typename T1, typename T2> int count(T1 const* beg, T1 const* end, T2 val); 

  /*!*****************************************************************************
  \brief
    Display all the data inside the half-open range array

  \param beg
    Pointer to the start of the half-open range array
  \param end
    Pointer to the end of the half-open range array
  *******************************************************************************/ 
  template <typename T> void display(T const* beg, T const* end); 

  /*!*****************************************************************************
  \brief
    Checks if the half-open range array contains the same dataset compared to src
      
  \param beg
    Pointer to the start of the half-open range array
  \param end
    Pointer to the end of the half-open range array
  \param src
    Pointer to the start of src

  \return
    True if half-open range array contains the same dataset as src, else false
  *******************************************************************************/ 
  template <typename T1, typename T2> bool equal(T1 const* beg, T1 const* end, T2 const* src); 

  /*!*****************************************************************************
  \brief
    Checks if the half-open range array contains the same dataset compared to src
      
  \param beg
    Pointer to the start of the half-open range array
  \param end
    Pointer to the end of the half-open range array
  \param src
    Pointer to the start of src

  \return
    True if half-open range array contains the same dataset as src, else false
  *******************************************************************************/ 
  template <typename T> T const* find(T const* beg, T const* end, T val);

  /*!*****************************************************************************
  \brief
    Find the first instance of val in the 

  \param beg
    Pointer to the start of the half-open range array
  \param end
    Pointer to the end of the half-open range array
  \param val
    Value to find inside the half-open range array
  *******************************************************************************/ 
  template <typename T> T* find(T* beg, T* end, T val);

  /*!*****************************************************************************
  \brief
    Fill in the entire half-open range array with val
      
  \param beg
    Pointer to the start of the half-open range array
  \param end
    Pointer to the end of the half-open range array
  \param val
    Value to be filled inside the half-open range array
  *******************************************************************************/ 
  template <typename T1, typename T2> void fill(T1* beg, T1* end, T2 val);
  
  /*!*****************************************************************************
  \brief
    Iterates through the half-open range array and returns the pointer the biggest
    element
      
  \param beg
    Pointer to the start of the half-open range array
  \param end
    Pointer to the end of the half-open range array

  \return
    The pointer to the biggest element in the half-open range array
  *******************************************************************************/ 
  template <typename T> T* max_element(T* beg, T* end);

  /*!*****************************************************************************
  \brief
    Iterates through the half-open range array and returns the pointer the smallest
    element
      
  \param beg
    Pointer to the start of the half-open range array
  \param end
    Pointer to the end of the half-open range array

  \return
    The pointer to the smallest element in the half-open range array
  *******************************************************************************/  
  template <typename T> T* min_element(T* beg, T* end);

  /*!*****************************************************************************
  \brief
    Remove all instances of val in the half-open range
      
  \param beg
    Pointer to the start of the half-open range array
  \param end
    Pointer to the end of the half-open range array
  \param val
    Value to be removed from the half-open range array

  \return
    A new pointer to the end of the half-open range array
  *******************************************************************************/ 
  template <typename T> T* remove(T* beg, T* end, T val);
  
  /*!*****************************************************************************
  \brief
    Replace all instances of oldVal to newVal inside the half-open range
    
  \param beg
    Pointer to the start of the half-open range array
  \param end
    Pointer to the end of the half-open range array
  \param oldVal
    Old value to be replaced
  \param newVal
    New value to be replace with
  *******************************************************************************/ 
  template <typename T1, typename T2> void replace(T1* beg, T1* end, T2 oldVal, T2 newVal);

  /*!*****************************************************************************
  \brief
    The total sum of values of the half-open range array

  \param beg
    Pointer to the start of the half-open range array
  \param end
    Pointer to the end of the half-open range array

  \return
    The total sum of values from beg to one before end
  *******************************************************************************/  
  template <typename T> T sum(T const* beg, T const* end);
  
  /*!*****************************************************************************
  \brief
    Swaps two objects in place.

  \param lhs
    Reference to the first object to swap.
  \param rhs
    Reference to the second object to swap.
  *******************************************************************************/
  template <typename T> void swap(T &lhs, T &rhs);

  /*!*****************************************************************************
  \brief
    Swaps an half-open range of array to another array in place.

  \param beg
    Pointer to the start of the half-open range array
  \param end
    Pointer to the end of the half-open range array
  \param src
    Pointer to the start of the second array
  *******************************************************************************/
  template <typename T> void swap_ranges(T* beg, T* end, T* src);

  template <typename T>
  T* copy(T const* beg, T const* end, T* dst)
  {
    T* ptr = dst;
    while(beg < end)
      *ptr++ = *beg++;
    return ptr;
  }

  template <typename T1, typename T2>
  int count(T1 const* beg, T1 const* end, T2 val)
  {
    int count{ 0 };
    while(beg < end)
    {
      if(*beg++ == val)
        ++count;
    }
    return count;
  }
  
  template <typename T>
  void display(T const* beg, T const* end)
  {
    while(beg < end)
    {
      std::cout << *beg;
      if(beg++ != end - 1)
        std::cout << ", ";
    }
    std::cout << std::endl;
  }
  
  template <typename T1, typename T2>
  bool equal(T1 const* beg, T1 const* end, T2 const* src)
  {
    while(beg < end)
    {
      if(*beg++ != *src++)
        return false;
    }
    return true;
  }
  
  template <typename T>
  T const* find(T const* beg, T const* end, T val)
  {
    while(beg < end)
    {
      if(*beg == val)
        return beg;
      ++beg;
    }
    return end;
  }
  
  template <typename T>
  T* find(T* beg, T* end, T val)
  {
    while(beg < end)
    {
      if(*beg == val)
        return beg;
      ++beg;
    }
    return end;
  }
  
  template <typename T1, typename T2>
  void fill(T1* beg, T1* end, T2 val)
  {
    while(beg < end)
      *beg++ = val;
  }
  
  template <typename T>
  T* max_element(T* beg, T* end)
  {
    T* max = beg++;
    while(beg < end)
    {
      if(*max < *beg++)
        max = --beg;
    }
    return max;
  }
  
  template <typename T>
  T* min_element(T* beg, T* end)
  {
    T* min = beg++;
    while(beg < end)
    {
      if(*beg++ < *min)
        min = --beg;
    }
    return min;
  }
  
  template <typename T>
  T* remove(T* beg, T* end, T val)
  {
    size_t num = static_cast<size_t>(count(beg, end, val)), size = (end - beg) - num, i{ 0 };
    for(T* ptr = beg; i < size; ++beg)
    {
      if(*beg != val)
        *(ptr + i++) = *beg;
    }
    return end - num;
  }
  
  template <typename T1, typename T2>
  void replace(T1* beg, T1* end, T2 oldVal, T2 newVal)
  {
    while(beg < end)
    {
      if(*beg == oldVal)
        *beg = newVal;
      ++beg;
    }
  }

  template <typename T>
  T sum(T const* beg, T const* end)
  {
    T sum{};
    while(beg < end)
      sum += *beg++;
    return sum;
  }
  
  template <typename T> void swap(T &lhs, T &rhs)
  {
    T temp{ lhs };
    lhs = rhs;
    rhs = temp;
  }

  template <typename T>
  void swap_ranges(T* beg, T* end, T* src)
  {
    while(beg < end)
      swap(*beg++, *src++);
  }
}

#endif
//-------------------------------------------------------------------------
