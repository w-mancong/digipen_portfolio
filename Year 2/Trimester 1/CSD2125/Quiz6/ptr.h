// TODO: Provide file documentation header

#include <iostream> // std::ostream

// TODO: Don't include any other C and C++ standard library headers!!!

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
        ~Ptr() { std::cout << __PRETTY_FUNCTION__ << std::endl;  delete p; } // DO NOT AMEND!!!
        T* get() const { return p; }    // DO NOT AMEND!!!

        Ptr(Ptr const &rhs);

        reference operator*() const;

        pointer operator->() const;

        Ptr &operator=(Ptr const &rhs);

        template <typename U>
        Ptr(Ptr<U> const& rhs);

        template <typename U>
        Ptr &operator=(Ptr<U> const &rhs);

    private:
        T *p; // DO NOT AMEND!!!
    };

    template <typename T>
    Ptr<T>::Ptr(Ptr const& rhs)
    {

    }

    template <typename T>
    typename Ptr<T>::reference Ptr<T>::operator*() const
    {

    }

    template <typename T>
    typename Ptr<T>::pointer Ptr<T>::operator->() const
    {

    }

    template <typename T>
    Ptr<T>& Ptr<T>::operator=(Ptr const& rhs)
    {

    }

    template <typename T>
    template <typename U>
    Ptr<T>::Ptr(Ptr<U> const& rhs)
    {

    }

    template <typename T>
    template <typename U>
    Ptr<T>& Ptr<T>::operator=(Ptr<U> const& rhs)
    {

    }   
} // end namespace HLP3

#endif // #ifndef PTR_H