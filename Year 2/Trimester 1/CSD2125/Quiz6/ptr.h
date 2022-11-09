/*!*****************************************************************************
\file ptr.h
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Quiz 6
\date 9-11-2022
\brief
Wrapper class for pointers
*******************************************************************************/
#include <iostream> // std::ostream

#ifndef PTR_H
#define PTR_H

namespace HLP3 
{
    // partially defined class template Ptr
    template <typename T>
    class Ptr 
    {
    public:
        using value_type        = T;
        using size_type         = size_t;
        using reference         = value_type &;
        using const_reference   = value_type const &;
        using pointer           = value_type *;
        using const_pointer     = value_type const *;
        using iterator          = pointer;
        using const_iterator    = const_pointer;

        Ptr()                 = delete; // DO NOT AMEND!!!
        Ptr(Ptr&&)            = delete; // DO NOT AMEND!!!
        Ptr& operator=(Ptr&&) = delete; // DO NOT AMEND!!!
        explicit Ptr(T* _p) : p{_p} {}  // DO NOT AMEND!!!
        ~Ptr() { std::cout << __PRETTY_FUNCTION__ << std::endl; delete p; } // DO NOT AMEND!!!
        T* get() const { return p; }    // DO NOT AMEND!!!

        /*!*****************************************************************************
            \brief Copy constructor that construct a new pointer and store whatever value
            rhs stores
        *******************************************************************************/
        Ptr(Ptr const &rhs);

        /*!*****************************************************************************
            \brief Returns a reference to value_type
        *******************************************************************************/
        reference operator*() const;

        /*!*****************************************************************************
            \brief Return a pointer to value_type
        *******************************************************************************/
        pointer operator->() const;

        /*!*****************************************************************************
            \brief Assign value_type in rhs to this object

            \return A reference to Ptr obj
        *******************************************************************************/
        Ptr &operator=(Ptr const &rhs);

        /*!*****************************************************************************
            \brief Copy constructor that constructs an object of template type U and 
            static cast it to value_type of template T
        *******************************************************************************/
        template <typename U>
        Ptr(Ptr<U> const& rhs);

        /*!*****************************************************************************
            \brief Assigning a value_type of template U to value_type of template T
        *******************************************************************************/
        template <typename U>
        Ptr &operator=(Ptr<U> const &rhs);

    private:
        T *p; // DO NOT AMEND!!!
    };

    template <typename T>
    Ptr<T>::Ptr(Ptr const& rhs) : p{ new value_type{ *rhs } }
    {

    }

    template <typename T>
    typename Ptr<T>::reference Ptr<T>::operator*() const
    {
        return *p;
    }

    template <typename T>
    typename Ptr<T>::pointer Ptr<T>::operator->() const
    {
        return p;
    }

    template <typename T>
    Ptr<T>& Ptr<T>::operator=(Ptr const& rhs)
    {
        *p = *rhs;
        return *this;
    }

    template <typename T>
    template <typename U>
    Ptr<T>::Ptr(Ptr<U> const& rhs) : p{ new value_type{ static_cast<T>(*rhs) } }
    {

    }

    template <typename T>
    template <typename U>
    Ptr<T>& Ptr<T>::operator=(Ptr<U> const& rhs)
    {
        *p = static_cast<T>(*rhs);
        return *this;
    }   
} // end namespace HLP3

#endif // #ifndef PTR_H