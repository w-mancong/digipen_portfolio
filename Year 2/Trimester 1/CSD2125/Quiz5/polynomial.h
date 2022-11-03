// Provide file documentation header
// Don't include any C and C++ standard library headers!!!
// Remember, this file is incomplete and you must provide missing details/features.

#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <iostream> // std::ostream

namespace HLP3 
{
    // declare class template Polynomial
    template <typename T1, size_t N>
    class Polynomial
    {
    public:
        using value_type = T1;
        using reference = value_type &;
        using const_reference = value_type const &;
        using size_type = size_t;

        Polynomial(void);
        ~Polynomial(void);
        template <typename T2>
        Polynomial(Polynomial<T2, N> const &rhs);
        template <typename T2>
        Polynomial &operator=(Polynomial<T2, N> const &rhs);
        value_type operator()(int a);

        template <size_t M>
        Polynomial<T1, N + M> operator*(Polynomial<T1, M> const &rhs);

        reference operator[](size_type index);
        const_reference operator[](size_type index) const;

    private:
        value_type *values{ nullptr };
    };
}

#include "polynomial.tpp"

#endif
