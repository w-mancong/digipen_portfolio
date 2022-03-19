/*!*****************************************************************************
\file matrix.hpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\date 16-03-2022
\brief
This file contain function declarations & definition for a templated matrix
class
*******************************************************************************/
#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <algorithm>
#include "matrix_exception.h"

namespace ManCong
{
    class matrix
    {
    public:
        using value_type        = float;
        using reference         = value_type&;
        using const_reference   = value_type const&;
        using size_type         = long long;
    public:
        matrix(size_type R, size_type C);
        ~matrix(void);
        matrix(matrix const& rhs);
        matrix&         operator=(matrix const& rhs);

        reference       operator()(size_type row, size_type col);
        const_reference const& operator()(size_type row, size_type col) const;

        bool            operator==(matrix const& rhs) const;
        bool            operator!=(matrix const& rhs) const;

        matrix&         operator+=(matrix const& rhs);

        matrix&         operator-=(matrix const& rhs);

        matrix&         operator*=(matrix const& rhs);
        matrix&         operator*=(value_type rhs);

        size_type       Rows(void) const;
        size_type       Cols(void) const;

        matrix&         Transpose(void);
        matrix&         Inverse(void);
        void            Indentity(void);
        value_type      Determinant(void) const;

    private:
        value_type      Determinant(matrix const& mtx, size_type n) const;
        void            BarMatrix(matrix& dst, matrix const& src, size_type row, size_type col) const;
        const_reference cget(size_type row, size_type col) const;
        void            swap(matrix& rhs);
        void            swap(matrix& lhs, matrix& rhs);

        value_type *mtx;
        size_type R, C;
    };

    matrix operator+(matrix const& lhs, matrix const& rhs);

    matrix operator-(matrix const& lhs, matrix const& rhs);

    matrix operator*(matrix const& lhs, matrix const& rhs);

    matrix operator*(matrix const& lhs, typename matrix::value_type rhs);

    matrix operator*(typename matrix::value_type lhs, matrix const& rhs);

    std::ostream& operator<<(std::ostream& os, matrix const& rhs);
}
#endif