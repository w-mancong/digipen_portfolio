/*!*****************************************************************************
\file polynomial.h
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Quiz 5
\date 4-11-2022
\brief
Templated class that provides functionalities to declaring and calculating
polynomials
*******************************************************************************/
#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <iostream>

namespace HLP3 
{
    /*!*****************************************************************************
        \brief Templated class for Polynomial
    *******************************************************************************/
    template <typename T1, int N>
    class Polynomial
    {
    public:
        using value_type = T1;
        using reference = value_type &;
        using const_reference = value_type const &;
        using size_type = size_t;

        /*!*****************************************************************************
            \brief Default construtor: Makes a zero polynomial
        *******************************************************************************/
        Polynomial(void);

        /*!*****************************************************************************
            \brief Default destructor
        *******************************************************************************/
        ~Polynomial(void) = default;

        /*!*****************************************************************************
            \brief Copy constructor, allows for conversion between different data types
        *******************************************************************************/
        template <typename T2>
        Polynomial(Polynomial<T2, N> const &rhs);

        /*!*****************************************************************************
            \brief Copy assignment, allows for conversion between different data types
        *******************************************************************************/
        template <typename T2>
        Polynomial &operator=(Polynomial<T2, N> const &rhs);

        /*!*****************************************************************************
            \brief Evaluates the expression p(x) where x = a

            \param [in] a: The value to be substituted with x
        *******************************************************************************/
        value_type operator()(int a);

        /*!*****************************************************************************
            \brief Multiplication of two polynomials -> p(x) * q(x)

            \param [in] rhs: Polynomial q(x)
        *******************************************************************************/
        template <int M>
        Polynomial<T1, N + M> operator*(Polynomial<T1, M> const &rhs);

        /*!*****************************************************************************
            \brief To retrieve a reference to values

            \param [in] index: Index of the array
        *******************************************************************************/
        reference operator[](size_type index);

        /*!*****************************************************************************
            \brief To retrieve a const reference to values

            \param [in] index: Index of the array
        *******************************************************************************/
        const_reference operator[](size_type index) const;

    private:
        value_type values[N + 1];

        /*!*****************************************************************************
            \brief Swap values internally

            \param [in] rhs: To have values be swapped with this object
        *******************************************************************************/
        void swap(Polynomial<T1, N> &rhs);

        /*!*****************************************************************************
            \brief Power function

            \param [in] base: Base value
            \param [in] exponent: Exponent of the base
        *******************************************************************************/
        value_type pow(value_type base, size_type exponent);
    };
}

#include "polynomial.tpp"

#endif
