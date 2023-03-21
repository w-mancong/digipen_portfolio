/*!*****************************************************************************
\file matrix.hpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Assignment 8
\date 16-03-2022
\brief
This file contain function declarations & definition for a templated matrix
class
*******************************************************************************/
#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <algorithm>
#include <iostream>

namespace hlp2
{
    template <typename T>
    class matrix
    {
    public:
        /*!*****************************************************************************
        \brief
            Default constructor of matrix class
        *******************************************************************************/ 
        matrix(unsigned int rows, unsigned int cols);

        /*!*****************************************************************************
        \brief
            Destructor for matrix class
        *******************************************************************************/
        ~matrix(void);

        /*!*****************************************************************************
        \brief
            Copy constructor for matrix class
        \param [in] rhs
            matrix class to copy it's data into
        *******************************************************************************/
        matrix(matrix const &rhs);

        /*!*****************************************************************************
        \brief
            Overloaded copy assignment operator
        \param [in] rhs
            matrix class to copy it's data into
        \return
            A reference to this class
        *******************************************************************************/
        matrix &operator=(matrix const &rhs);

        /*!*****************************************************************************
        \brief
            Overloaded subscript operator for accessing and modifying members
        \param [in] index
            Index of the data inside the std::vector
        \return
            A reference to the std::vector at position index
        *******************************************************************************/
        std::vector<T> &operator[](size_t index);

        /*!*****************************************************************************
        \brief
            Overloaded subscript operator for accessing members
        \param [in] index
            Index of the data inside the std::vector
        \return 
            A const reference to the std::vector at position index
        *******************************************************************************/
        std::vector<T> const &operator[](size_t index) const;

        /*!*****************************************************************************
        \brief
            Checks if the two matrices are the same
        \param [in] rhs
            Matrix to be check with
        \return 
            True if the two matrices are the same
        *******************************************************************************/
        bool operator==(matrix const &rhs) const;

        /*!*****************************************************************************
        \brief
            Checks if the two matrices are not the same
        \param [in] rhs
            Matrix to be check with
        \return
            True if the two matrices are not the same
        *******************************************************************************/
        bool operator!=(matrix const &rhs) const;

        /*!*****************************************************************************
        \brief
            Add two matrices together
        \param [in] rhs
            Matrix to be added with
        \return 
            A reference to this class
        *******************************************************************************/
        matrix &operator+=(matrix const &rhs);

        /*!*****************************************************************************
        \brief
            Subtract two matrices
        \param [in] rhs
            Matrix to be subtracted with
        \return
            A reference to this class
        *******************************************************************************/
        matrix &operator-=(matrix const &rhs);

        /*!*****************************************************************************
        \brief
            Matrix multiplication with another matrix
        \param [in] rhs
            Matrix to be multiplied with
        \return 
            A reference to this class
        *******************************************************************************/
        matrix &operator*=(matrix const &rhs);

        /*!*****************************************************************************
        \brief
            Matrix multiplication with a scalar
        \param [in] rhs
            Scalar multiple to be multiplied with
        \return 
            A reference to this calss
        *******************************************************************************/
        matrix &operator*=(T const &rhs);

        /*!*****************************************************************************
        \brief
            Returns total number of rows in the matrix
        \return
            Total number of rows
        *******************************************************************************/
        size_t Rows(void) const;

        /*!*****************************************************************************
        \brief
            Returns total number of columns in the matrix
        \return
            Total number of columns
        *******************************************************************************/
        size_t Cols(void) const;

    private:
        std::vector<std::vector<T>> mtx;
    };

    /*!*****************************************************************************
    \brief
        Addition of two matrices
    \param [in] lhs
        First matrix to be added
    \param [in] rhs
        Second matrix to be added
    \return
        A copy of the matrix after adding lhs and rhs together
    *******************************************************************************/
    template <typename T>
    matrix<T> operator+(matrix<T> const &lhs, matrix<T> const &rhs);

    /*!*****************************************************************************
    \brief
        Subtration of two matrices
    \param [in] lhs
        First matrix to be subtracted
    \param [in] rhs
        Secodn matrix to be subtracted
    \return
        A copy of the matrix after subtracting lhs from rhs 
    *******************************************************************************/
    template <typename T>
    matrix<T> operator-(matrix<T> const &lhs, matrix<T> const &rhs);

    /*!*****************************************************************************
    \brief
        Matrix multiplication with another matrix
    \param [in] lhs
        First matrix for the multiplication
    \param [in] rhs
        Second matrix for the multiplication
    \return 
        A copy of the matrix after the matrix multiplication
    *******************************************************************************/
    template <typename T>
    matrix<T> operator*(matrix<T> const &lhs, matrix<T> const &rhs);

    /*!*****************************************************************************
    \brief
        Matrix multiplication with a scalar
    \param [in] lhs
        Matrix for multiplication
    \param [in] rhs
        Scalar quantity for multiplication
    \return
        A copy of the matrix after multiplying it with a scalar
    *******************************************************************************/
    template <typename T>
    matrix<T> operator*(matrix<T> const &lhs, T const &rhs);

    /*!*****************************************************************************
    \brief
        Matrix multiplcation with a scalar
    \param [in] lhs
        Scalar quantity for multiplication
    \param [in] rhs
        Matrix for multiplication
    \return
        A copy of the matrix after multiplying it with a scalar
    *******************************************************************************/
    template <typename T>
    matrix<T> operator*(T const &lhs, matrix<T> const &rhs);

    /*!*****************************************************************************
    \brief
        Overloaded left bit-shift operator to write the data into ostream
    \param [in, out] os
        A reference to ostream
    \param [in] rhs
        The matrix data to print it's data from
    \return
        A reference to ostream
    *******************************************************************************/
    template <typename T>
    std::ostream &operator<<(std::ostream &os, matrix<T> const &rhs);

    template <typename T>
    matrix<T>::matrix(unsigned int rows, unsigned int cols)
    {
        mtx.resize(rows);
        for (unsigned int i = 0; i < rows; ++i)
            mtx[i].resize(cols);
        for (unsigned int i = 0; i < rows; ++i)
        {
            for (unsigned int j = 0; j < cols; ++j)
                mtx[i][j] = {};
        }
    }

    template <typename T>
    matrix<T>::~matrix(void) {}

    template <typename T>
    matrix<T>::matrix(matrix const &rhs)
    {
        unsigned int rows = rhs.Rows(), cols = rhs.Cols();
        mtx.resize(rows);
        for (unsigned int i = 0; i < rows; ++i)
            mtx[i].resize(cols);
        for (unsigned int i = 0; i < rows; ++i)
        {
            for (unsigned int j = 0; j < cols; ++j)
                mtx[i][j] = rhs[i][j];
        }
    }

    template <typename T>
    matrix<T> &matrix<T>::operator=(matrix const &rhs)
    {
        matrix<T> tmp{rhs};
        std::swap(mtx, tmp.mtx);
        return *this;
    }

    template <typename T>
    std::vector<T> &matrix<T>::operator[](size_t index)
    {
        return const_cast<std::vector<T> &>(static_cast<matrix<T> const &>(*this)[index]);
    }

    template <typename T>
    std::vector<T> const &matrix<T>::operator[](size_t index) const
    {
        return mtx[index];
    }

    template <typename T>
    bool matrix<T>::operator==(matrix const &rhs) const
    {
        unsigned int const rows = Rows(), cols = Cols();
        for (unsigned int i = 0; i < rows; ++i)
        {
            for (unsigned int j = 0; j < cols; ++j)
            {
                if (mtx[i][j] != rhs.mtx[i][j])
                    return false;
            }
        }
        return true;
    }

    template <typename T>
    bool matrix<T>::operator!=(matrix const &rhs) const
    {
        return !(*this == rhs);
    }

    template <typename T>
    matrix<T> &matrix<T>::operator+=(matrix const &rhs)
    {
        unsigned int const rows = Rows(), cols = Cols();
        for (unsigned int i = 0; i < rows; ++i)
        {
            for (unsigned int j = 0; j < cols; ++j)
                mtx[i][j] += rhs.mtx[i][j];
        }
        return *this;
    }

    template <typename T>
    matrix<T> &matrix<T>::operator-=(matrix const &rhs)
    {
        unsigned int const rows = Rows(), cols = Cols();
        for (unsigned int i = 0; i < rows; ++i)
        {
            for (unsigned int j = 0; j < cols; ++j)
                mtx[i][j] -= rhs.mtx[i][j];
        }
        return *this;
    }

    template <typename T>
    matrix<T> &matrix<T>::operator*=(matrix const &rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    template <typename T>
    matrix<T> &matrix<T>::operator*=(T const &rhs)
    {
        unsigned int const rows = Rows(), cols = Cols();
        for (unsigned int i = 0; i < rows; ++i)
        {
            for (unsigned int j = 0; j < cols; ++j)
                mtx[i][j] *= rhs;
        }
        return *this;
    }

    template <typename T>
    size_t matrix<T>::Rows(void) const
    {
        return mtx.size();
    }

    template <typename T>
    size_t matrix<T>::Cols(void) const
    {
        return mtx[0].size();
    }

    template <typename T>
    matrix<T> operator+(matrix<T> const &lhs, matrix<T> const &rhs)
    {
        matrix<T> tmp{lhs};
        tmp += rhs;
        return tmp;
    }

    template <typename T>
    matrix<T> operator-(matrix<T> const &lhs, matrix<T> const &rhs)
    {
        matrix<T> tmp{lhs};
        tmp -= rhs;
        return tmp;
    }

    template <typename T>
    matrix<T> operator*(matrix<T> const &lhs, matrix<T> const &rhs)
    {
        const unsigned int l_cols = lhs.Cols(), l_rows = lhs.Rows(), r_cols = rhs.Cols();
        const unsigned int final_size = l_rows * r_cols;
        unsigned int curr_size = 0;
        unsigned int k = 0, l = 0;
        matrix<T> tmp(l_rows, r_cols);

        while (curr_size < final_size)
        {
            for (unsigned int i = 0; i < r_cols; ++i)
            {
                T sum = {};
                for (unsigned int j = 0; j < l_cols; ++j)
                    sum += lhs[l][j] * rhs[j][k];
                tmp[l][k++] = sum, ++curr_size;
            }
            if (!(curr_size % r_cols))
                ++l, k = 0;
        }
        return tmp;
    }

    template <typename T>
    matrix<T> operator*(matrix<T> const &lhs, T const &rhs)
    {
        matrix<T> tmp{lhs};
        return tmp *= rhs;
    }

    template <typename T>
    matrix<T> operator*(T const &lhs, matrix<T> const &rhs)
    {
        matrix<T> tmp{rhs};
        return tmp *= lhs;
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &os, hlp2::matrix<T> const &rhs)
    {
        const unsigned int rows = rhs.Rows(), cols = rhs.Cols();
        for (unsigned int i = 0; i < rows; ++i)
        {
            for (unsigned int j = 0; j < cols; ++j)
                os << rhs[i][j] << (j + 1 == cols ? "" : " ");
            os << std::endl;
        }
        return os;
    }
}
#endif