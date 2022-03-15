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

#include <algorithm>
#include <iostream>

namespace ManCong
{
    template <size_t R, size_t C>
    class matrix
    {
    public:
        matrix(void);
        ~matrix(void);
        matrix(matrix const& rhs);
        matrix& operator=(matrix const& rhs);

        float(&operator[](size_t index))[C];
        float cget(size_t row, size_t col) const;

        bool operator==(matrix const& rhs) const;
        bool operator!=(matrix const& rhs) const;

        matrix& operator+=(matrix const& rhs);

        matrix& operator-=(matrix const& rhs);

        matrix<R, C>& operator*=(matrix<C, C> const& rhs);
        matrix& operator*=(float rhs);

        size_t Rows(void) const;
        size_t Cols(void) const;

    private:
        float mtx[R][C];
    };

    template <size_t R, size_t C>
    matrix<R, C> operator+(matrix<R, C> const& lhs, matrix<R, C> const& rhs);

    template <size_t R, size_t C>
    matrix<R, C> operator-(matrix<R, C> const& lhs, matrix<R, C> const& rhs);

    template <size_t R, size_t CR, size_t C>
    matrix<R, C> operator*(matrix<R, CR> const& lhs, matrix<CR, C> const& rhs);

    template <size_t R, size_t C>
    matrix<R, C> operator*(matrix<R, C> const& lhs, float rhs);

    template <size_t R, size_t C>
    matrix<R, C> operator*(float lhs, matrix<R, C> const& rhs);

    template <size_t R, size_t C>
    std::ostream& operator<<(std::ostream& os, matrix<R, C> const& rhs);

    template <size_t R, size_t C>
    matrix<R, C>::matrix(void)
    {
        memset(mtx, 0, sizeof(mtx));
    }

    template <size_t R, size_t C>
    matrix<R, C>::~matrix(void) {}

    template <size_t R, size_t C>
    matrix<R, C>::matrix(matrix const& rhs)
    {
        for (size_t i = 0; i < R; ++i)
        {
            for (size_t j = 0; j < C; ++j)
                mtx[i][j] = rhs.cget(i, j);
        }
    }

    template <size_t R, size_t C>
    matrix<R, C>& matrix<R, C>::operator=(matrix const& rhs)
    {
        matrix<R, C> tmp{ rhs };
        std::swap(mtx, tmp.mtx);
        return *this;
    }

    template <size_t R, size_t C>
    float(&matrix<R, C>::operator[](size_t index))[C]
    {
        return mtx[index];
    }

    template <size_t R, size_t C>
    float matrix<R, C>::cget(size_t row, size_t col) const
    {
        return mtx[row][col];
    }

    template <size_t R, size_t C>
    bool matrix<R, C>::operator==(matrix const& rhs) const
    {
        for (size_t i = 0; i < R; ++i)
        {
            for (size_t j = 0; j < C; ++j)
            {
                if (mtx[i][j] != rhs[i][j])
                    return false;
            }
        }
        return true;
    }

    template <size_t R, size_t C>
    bool matrix<R, C>::operator!=(matrix const& rhs) const
    {
        return !(*this == rhs);
    }

    template <size_t R, size_t C>
    matrix<R, C>& matrix<R, C>::operator+=(matrix const& rhs)
    {
        for (size_t i = 0; i < R; ++i)
        {
            for (size_t j = 0; j < C; ++j)
                mtx[i][j] += rhs[i][j];
        }
        return *this;
    }

    template <size_t R, size_t C>
    matrix<R, C>& matrix<R, C>::operator-=(matrix const& rhs)
    {
        for (size_t i = 0; i < R; ++i)
        {
            for (size_t j = 0; j < C; ++j)
                mtx[i][j] -= rhs[i][j];
        }
        return *this;
    }

    template <size_t R, size_t C>
    matrix<R, C>& matrix<R, C>::operator*=(matrix<C, C> const& rhs)
    {
        const size_t final_size = R * C;
        size_t curr_size = 0, k = 0, l = 0;
        matrix<R, C> tmp;
        while (curr_size < final_size)
        {
            for (size_t i = 0; i < C; ++i)
            {
                float sum = {};
                for (size_t j = 0; j < C; ++j)
                    sum += mtx[l][j] * rhs.cget(j, k);
                tmp[l][k++] = sum, ++curr_size;
            }
            // condition for me to know that im done with the current row
            if (!(curr_size % C))
                ++l, k = 0;
        }
        return (*this = tmp);
    }

    template <size_t R, size_t C>
    matrix<R, C>& matrix<R, C>::operator*=(float rhs)
    {
        for (size_t i = 0; i < R; ++i)
        {
            for (size_t j = 0; j < C; ++j)
                mtx[i][j] *= rhs;
        }
    }

    template <size_t R, size_t C>
    size_t matrix<R, C>::Rows(void) const
    {
        return R;
    }

    template <size_t R, size_t C>
    size_t matrix<R, C>::Cols(void) const
    {
        return C;
    }

    template <size_t R, size_t C>
    matrix<R, C> operator+(matrix<R, C> const& lhs, matrix<R, C> const& rhs)
    {
        matrix<R, C> tmp{ lhs };
        tmp += rhs;
        return tmp;
    }

    template <size_t R, size_t C>
    matrix<R, C> operator-(matrix<R, C> const& lhs, matrix<R, C> const& rhs)
    {
        matrix<R, C> tmp{ lhs };
        tmp -= rhs;
        return tmp;
    }

    template <size_t R, size_t CR, size_t C>
    matrix<R, C> operator*(matrix<R, CR> const& lhs, matrix<CR, C> const& rhs)
    {
        const size_t final_size = R * C;
        size_t curr_size = 0, k = 0, l = 0;
        matrix<R, C> tmp;
        while (curr_size < final_size)
        {
            for (size_t i = 0; i < C; ++i)
            {
                float sum = {};
                for (size_t j = 0; j < CR; ++j)
                    sum += lhs.cget(l, j) * rhs.cget(j, k);
                tmp[l][k++] = sum, ++curr_size;
            }
            // condition for me to know that im done with the current row
            if (!(curr_size % C))
                ++l, k = 0;
        }
        return tmp;
    }

    template <size_t R, size_t C>
    matrix<R, C> operator*(matrix<R, C> const& lhs, float rhs)
    {
        matrix<R, C> tmp{ lhs };
        return tmp *= rhs;
    }

    template <size_t R, size_t C>
    matrix<R, C> operator*(float lhs, matrix<R, C> const& rhs)
    {
        matrix<R, C> tmp{ rhs };
        return tmp *= lhs;
    }

    template <size_t R, size_t C>
    std::ostream& operator<<(std::ostream& os, matrix<R, C> const& rhs)
    {
        for (size_t i = 0; i < R; ++i)
        {
            for (size_t j = 0; j < C; ++j)
                os << rhs.cget(i, j) << (j + 1 == C ? '\0' : ' ');
            os << std::endl;
        }
        return os;
    }
}
#endif