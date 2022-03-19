#include "matrix.hpp"

namespace ManCong
{
    matrix::matrix(size_type R, size_type C) : R{ R }, C{ C }
    {
        if (0 > R || 0 > C)
            throw InvalidDimension(R, C);
        // TODO : Change it to memory manager once i port to game engine
        mtx = new value_type[R * C] {};
        if (R == C)
            Indentity();
    }

    matrix::~matrix(void)
    {
        if (mtx)
        {
            delete[] mtx;
            mtx = nullptr;
        }
    }

    matrix::matrix(matrix const& rhs) : R{ rhs.R }, C{ rhs.C }
    {
        mtx = new value_type[R * C]{};
        for (size_t i = 0; i < R; ++i)
        {
            for (size_t j = 0; j < C; ++j)
                (*this)(i, j) = rhs(i, j);
        }
    }

    matrix& matrix::operator=(matrix const& rhs)
    {
        matrix tmp{ rhs };
        std::swap(mtx, tmp.mtx);
        return *this;
    }

    matrix::reference matrix::operator()(size_type row, size_type col)
    {
        return const_cast<reference>(cget(row, col));
    }

    typename matrix::const_reference matrix::operator()(size_type row, size_type col) const
    {
        return cget(row, col);
    }

    bool matrix::operator==(matrix const& rhs) const
    {
        for (size_type i = 0; i < R; ++i)
        {
            for (size_type j = 0; j < C; ++j)
            {
                if ((*this)(i, j) != rhs(i, j))
                    return false;
            }
        }
        return true;
    }

    bool matrix::operator!=(matrix const& rhs) const
    {
        return !(*this == rhs);
    }

    matrix& matrix::operator+=(matrix const& rhs)
    {
        if (R != rhs.R || C != rhs.C)
            throw IncompatibleMatrices("Addition", R, C, rhs.R, rhs.C);
        for (size_type i = 0; i < R; ++i)
        {
            for (size_type j = 0; j < C; ++j)
                (*this)(i, j) += rhs(i, j);
        }
        return *this;
    }

    matrix& matrix::operator-=(matrix const& rhs)
    {
        if (R != rhs.R || C != rhs.C)
            throw IncompatibleMatrices("Subtraction", R, C, rhs.R, rhs.C);
        for (size_type i = 0; i < R; ++i)
        {
            for (size_type j = 0; j < C; ++j)
                (*this)(i, j) -= rhs(i, j);
        }
        return *this;
    }

    matrix& matrix::operator*=(matrix const& rhs)
    {
        return *this = *this * rhs;
    }

    matrix& matrix::operator*=(value_type rhs)
    {
        for (size_type i = 0; i < R; ++i)
        {
            for (size_type j = 0; j < C; ++j)
                (*this)(i, j) *= rhs;
        }
        return *this;
    }

    typename matrix::size_type matrix::Rows(void) const
    {
        return R;
    }

    typename matrix::size_type matrix::Cols(void) const
    {
        return C;
    }

    matrix& matrix::Transpose(void)
    {
        matrix tmp{ C, R };
        for (size_t i = 0; i < R; ++i)
        {
            for (size_t j = 0; j < C; ++j)
                tmp(j, i) = (*this)(i, j);
        }
        swap(tmp);
        return *this;
    }

    matrix& matrix::Inverse(void)
    {
        if (R != C)
            throw InvalidDimension(R, C, "an inverse matrix. Must be a square matrix!");
        value_type det = Determinant(*this, R), flag = static_cast<value_type>(1);
        matrix inv(R, C), tmp(R - 1, C - 1);
        for (size_type i = 0; i < R; ++i)
        {
            for (size_type j = 0; j < C; ++j)
            {
                BarMatrix(tmp, *this, i, j);
                inv(i, j) = Determinant(tmp, tmp.R) * flag;
                flag *= -1.0f;
            }
        }
        inv *= static_cast<value_type>(1) / det;
        swap(inv);
        return *this;
    }

    void matrix::Indentity(void)
    {
        if (R != C)
            throw InvalidDimension(R, C, "an indentity matrix. Must be a square matrix!");
        for (size_type i = 0; i < R; ++i)
            (*this)(i, i) = static_cast<value_type>(1);
    }

    typename matrix::value_type matrix::Determinant(void) const
    {
        if (R != C)
            throw InvalidDimension(R, C, "finding determinant. Must be a square matrix!");
        return Determinant(*this, R);
    }

    typename matrix::value_type matrix::Determinant(matrix const& mtx, size_type n) const
    {
        if (n == 1)
            return mtx(0, 0);
        if (n == 2)
            return mtx(0, 0) * mtx(1, 1) - mtx(0, 1) * mtx(1, 0);
        value_type det = {};
        matrix tmp(n - 1, n - 1);
        for (size_type j = 0; j < mtx.C; ++j)
        {
            BarMatrix(tmp, mtx, 0, j);
            det += static_cast<value_type>(pow(-1.0, static_cast<double>(j))) * mtx(0, j) * Determinant(tmp, tmp.R);
        }
        return det;
    }

    void matrix::BarMatrix(matrix& dst, matrix const& src, size_type row, size_type col) const
    {
        for (size_type i = 0, r = 0; r < dst.R; ++i)
        {
            if (i == row)
                continue;
            for (size_type j = 0, c = 0; c < dst.C; ++j)
            {
                if (j == col)
                    continue;
                dst(r, c++) = src(i, j);
            }
            ++r;
        }
    }

    matrix::const_reference matrix::cget(size_type row, size_type col) const
    {
        if (0 > row || R <= row || 0 > col || C <= col)
            throw IndexOutOfBounds(row, R, col, C);
        return *(mtx + row * C + col);
    }

    void matrix::swap(matrix& rhs)
    {
        std::swap(mtx, rhs.mtx);
        std::swap(R, rhs.R);
        std::swap(C, rhs.C);
    }

    void matrix::swap(matrix& lhs, matrix& rhs)
    {
        std::swap(lhs.mtx, rhs.mtx);
        std::swap(lhs.R, rhs.R);
        std::swap(lhs.C, rhs.C);
    }

    matrix operator+(matrix const& lhs, matrix const& rhs)
    {
        matrix tmp{ lhs };
        tmp += rhs;
        return tmp;
    }

    matrix operator-(matrix const& lhs, matrix const& rhs)
    {
        matrix tmp{ lhs };
        tmp -= rhs;
        return tmp;
    }

    matrix operator*(matrix const& lhs, matrix const& rhs)
    {
        const typename matrix::size_type l_rows = lhs.Rows(), l_cols = lhs.Cols(), 
                                         r_rows = rhs.Rows(), r_cols = rhs.Cols();
        if (l_cols != r_rows)
            throw IncompatibleMatrices("Multiplication", l_rows, l_cols, r_rows, r_cols);
        matrix tmp(l_rows, r_cols);

        for (typename matrix::size_type i = 0; i < l_rows; ++i)
        {
            for (typename matrix::size_type j = 0; j < l_cols; ++j)
            {
                for (typename matrix::size_type k = 0; k < r_cols; ++k)
                    tmp(i, k) += lhs(i, j) * rhs(j, k);
            }
        }
        return tmp;
    }

    matrix operator*(matrix const& lhs, typename matrix::value_type rhs)
    {
        matrix tmp{ lhs };
        return tmp *= rhs;
    }

    matrix operator*(typename matrix::value_type lhs, matrix const& rhs)
    {
        matrix tmp{ rhs };
        return tmp *= lhs;
    }

    std::ostream& operator<<(std::ostream& os, matrix const& rhs)
    {
        const typename matrix::size_type R = rhs.Rows(), C = rhs.Cols();
        for (typename matrix::size_type i = 0; i < R; ++i)
        {
            for (typename matrix::size_type j = 0; j < C; ++j)
                os << rhs(i, j) << (j + 1 == C ? '\0' : ' ');
            os << std::endl;
        }
        return os;
    }
}