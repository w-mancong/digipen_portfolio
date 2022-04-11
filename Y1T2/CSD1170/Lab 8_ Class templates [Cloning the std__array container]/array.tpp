/*!*****************************************************************************
\file array.tpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 8
\date 13-03-2022
\brief
    This file provides defintions for the templated class Array
*******************************************************************************/

/*!*****************************************************************************
    \brief
        Default constructor of array class
*******************************************************************************/
template <typename T, size_t N>
hlp2::Array<T, N>::Array(void) {}

/*!*****************************************************************************
    \brief
        Conversion constructor that converts an initializer_list to an array class
    \param [in] rhs
        Initializer list that takes in value_type
*******************************************************************************/
template <typename T, size_t N>
hlp2::Array<T, N>::Array(std::initializer_list<value_type> const& rhs)
{
    pointer ptr = data;
    for (value_type const& vt : rhs)
        *ptr++ = vt;
}

/*!*****************************************************************************
    \brief
        Destructor
*******************************************************************************/
template <typename T, size_t N>
hlp2::Array<T, N>::~Array(void) {}

/*!*****************************************************************************
    \brief
        Copy constructor
    \param [in] rhs
        Array object to copy it's content over
*******************************************************************************/
template <typename T, size_t N>
hlp2::Array<T, N>::Array(Array const &rhs)
{
    pointer ptr = data;
    for (size_type i = 0; i < rhs.size(); ++i)
        *(ptr + i) = *(rhs.data + i);
}

/*!*****************************************************************************
    \brief
        Overloaded copy assignment operator
    \param [in] rhs
        Array object to copy it's content over
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::Array &hlp2::Array<T, N>::operator=(Array const &rhs)
{
    Array<value_type, N> temp{ rhs };
    swap(temp);
    return *this;
}

/*!*****************************************************************************
    \brief
        Get a pointer to the first element of the array
    \return
        The address of the first element
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::iterator hlp2::Array<T, N>::begin(void)
{
    return const_cast<iterator>(cbegin());
}

/*!*****************************************************************************
    \brief
        Get a const pointer to the first element of the array
    \return
        The address of the first element
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::const_iterator hlp2::Array<T, N>::begin(void) const
{
    return cbegin();
}

/*!*****************************************************************************
    \brief
        Get a pointer to one past the last element of the array
    \return
        The address to the end of the array
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::iterator hlp2::Array<T, N>::end(void)
{
    return const_cast<iterator>(cend());
}

/*!*****************************************************************************
    \brief
        Get a const pointer to one past the last element of the array
    \return
        The address to the end of the array
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::const_iterator hlp2::Array<T, N>::end(void) const
{
    return cend();
}

/*!*****************************************************************************
    \brief
        Get a const pointer to the first element of the array
    \return
        The address of the first element
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::const_iterator hlp2::Array<T, N>::cbegin(void) const
{
    return data;
}

/*!*****************************************************************************
    \brief
        Get a const pointer to one past the last element of the array
    \return
        The address to the end of the array
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::const_iterator hlp2::Array<T, N>::cend(void) const
{
    return data + size();
}

/*!*****************************************************************************
    \brief
        A reference to the first element of the array
    \return
        A reference to value_type
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::reference hlp2::Array<T, N>::front(void)
{
    return const_cast<reference>(const_cast<const_class_reference>(*this)[0]);
}

/*!*****************************************************************************
    \brief
        A const reference to the first element of the array
    \return
        A reference to value_type
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::const_reference hlp2::Array<T, N>::front(void) const
{
    return (*this)[0];
}

/*!*****************************************************************************
    \brief
        A reference to the last element of the array
    \return
        A reference to value_type
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::reference hlp2::Array<T, N>::back(void)
{
    return const_cast<reference>(const_cast<const_class_reference>(*this)[size() - 1]);
}

/*!*****************************************************************************
    \brief
        A const reference to the last element of the array
    \return
        A reference to value_type
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::const_reference hlp2::Array<T, N>::back(void) const
{
    return (*this)[size() - 1];
}

/*!*****************************************************************************
    \brief
        A reference to the element at the specified index
    \param [in] index
        Position of the element in the array
    \return
        A reference to value_type
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::reference hlp2::Array<T, N>::operator[](size_type index)
{
    return const_cast<reference>(const_cast<const_class_reference>(*this)[index]);
}

/*!*****************************************************************************
    \brief
        A const reference to the element at the specified index
    \param [in] index
        Position of the element in the array
    \return
        A reference to value_type
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::const_reference hlp2::Array<T, N>::operator[](size_type index) const
{
    return *(data + index);
}

/*!*****************************************************************************
    \brief
        Check if the array is empty
    \return
        True if the array is empty, else false
*******************************************************************************/
template <typename T, size_t N>
bool hlp2::Array<T, N>::empty(void) const
{
    return !size();
}

/*!*****************************************************************************
    \brief
        Total number of element containing in the array
    \return
        Number of elements in the array
*******************************************************************************/
template <typename T, size_t N>
typename hlp2::Array<T, N>::size_type hlp2::Array<T, N>::size(void) const
{
    return sizeof(data) / sizeof(*data);
}

/*!*****************************************************************************
    \brief
        Fill the entire array with val
    \param [in] val
        Data to fill the entire array with
*******************************************************************************/
template <typename T, size_t N>
void hlp2::Array<T, N>::fill(value_type const &val)
{
    for (iterator it = begin(); it < end(); ++it)
        *it = val;
}

/*!*****************************************************************************
    \brief
        Swap the data in this class with rhs
    \param [in] rhs
        The class to have it's data swap with
*******************************************************************************/
template <typename T, size_t N>
void hlp2::Array<T, N>::swap(Array &rhs)
{
    for (size_type i = 0; i < size(); ++i)
    {
        value_type temp{ rhs[i] };
        rhs[i]  = data[i];
        data[i] = temp;
    }
}