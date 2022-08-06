/*!*****************************************************************************
\file matrix.hpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Assignment 9
\date 17-03-2022
\brief
This file contain function declarations & definition for a templated matrix
class. On top of that, it includes additional classes to deal with exceptions
*******************************************************************************/
#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <exception>
#include <vector>
#include <algorithm>
#include <iostream>

namespace hlp2
{
    class InvalidDimension : public std::exception
    {
    public:
        /*!*****************************************************************************
        \brief
            Default constructor of InvalidDimension class
        \param [in] rows
            Number of rows of the invalid matrix class
        \param [in] cols
            Number of columns of the invalid matrix class
        *******************************************************************************/
        InvalidDimension(int rows, int cols);

        /*!*****************************************************************************
        \brief
            Destructor for InvalidDimension class
        *******************************************************************************/
        ~InvalidDimension(void) = default;

        /*!*****************************************************************************
        \brief
            Error message informing about InvalidDimension allocated for matrix class
        \return
            A message informing about InvalidDimension of matrix class created
        *******************************************************************************/
        virtual const char *what(void) const throw();

    private:
        char msg[256];
    };

    InvalidDimension::InvalidDimension(int rows, int cols)
    {
        const char *ex = "Invalid Dimension Exception: ", *err = "is an invalid dimension for ";
        if(0 > rows && 0 > cols)
            sprintf(msg, "%s%d and %d are invalid dimensions for rows and columns respectively", ex, rows, cols);
        else if (0 > rows)
            sprintf(msg, "%s%d %srows", ex, rows, err);
        else if (0 > cols)
            sprintf(msg, "%s%d %scolumns", ex, cols, err);            
    }
    const char* InvalidDimension::what(void) const throw()
    {
        return msg;
    }

    class IndexOutOfBounds : public std::exception
    {
    public:
        /*!*****************************************************************************
        \brief
            Default constructor of IndexOutOfBounds class
        \param [in] r_index
            Invalid row index
        *******************************************************************************/
        IndexOutOfBounds(int r_index);

        /*!*****************************************************************************
        \brief
            Destructor of IndexOutOfBounds class
        *******************************************************************************/
        ~IndexOutOfBounds(void) = default;

        /*!*****************************************************************************
        \brief
            Error message informing about Index being out of bound
        \return
            A message informing about index being out of range
        *******************************************************************************/
        virtual const char *what(void) const throw();

    private:
        char msg[256];
    };

    IndexOutOfBounds::IndexOutOfBounds(int r_index)
    {
        sprintf(msg, "Index Out Of Bounds Exception: %d is an invalid index for rows", r_index);
    }

    const char *IndexOutOfBounds::what(void) const throw()
    {
        return msg;
    }

    class IncompatibleMatrices : public std::exception
    {
    public:
        /*!*****************************************************************************
        \brief
            Default constructor of IncompatibleMatrices class
        \param [in] operation
            What kind of mathematic operation that had failed to excecute
        \param [in] l_rows
            LHS number of Rows
        \param [in] l_cols
            LHS number of Columns
        \param [in] r_rows
            RHS number of Rows
        \param [in] r_cols
            RHS number of Columns
        *******************************************************************************/
        IncompatibleMatrices(const char *operation, int l_rows, int l_cols, int r_rows, int r_cols);

        /*!*****************************************************************************
        \brief
            Destructor for IncompatibleMatrices class
        *******************************************************************************/
        ~IncompatibleMatrices(void) = default;

        /*!*****************************************************************************
        \brief
            Error message informing about mathematic arithmetic for matrix is invalid
        \return
            A message informing about invalid size of matrices doing mathematic arithmetic
        *******************************************************************************/
        virtual const char *what(void) const throw();

    private:
        char msg[256];
    };

    IncompatibleMatrices::IncompatibleMatrices(const char *operation, int l_rows, int l_cols, int r_rows, int r_cols)
    {
        sprintf(msg, "Incompatible Matrices Exception: %s of LHS matrix with dimensions %d X %d and RHS matrix with dimensions %d X %d is undefined", operation, l_rows, l_cols, r_rows, r_cols);
    }

    const char* IncompatibleMatrices::what(void) const throw()
    {
        return msg;
    }

    template <typename T>
    class matrix
    {
    public:
        using size_type  = int;
        using value_type = T;

    public:
        /*!*****************************************************************************
        \brief
            Default constructor of matrix class
        *******************************************************************************/
        matrix(size_type rows, size_type cols);

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
        std::vector<T> &operator[](size_type index);

        /*!*****************************************************************************
        \brief
            Overloaded subscript operator for accessing members
        \param [in] index
            Index of the data inside the std::vector
        \return 
            A const reference to the std::vector at position index
        *******************************************************************************/
        std::vector<T> const &operator[](size_type index) const;

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
        size_type Rows(void) const;

        /*!*****************************************************************************
        \brief
            Returns total number of columns in the matrix
        \return
            Total number of columns
        *******************************************************************************/
        size_type Cols(void) const;

    private:
        std::vector<std::vector<value_type>> mtx;
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
    matrix<T>::matrix(size_type rows, size_type cols)
    {
        if(0 > rows || 0 > cols)
            throw InvalidDimension(rows, cols);
        
        mtx.resize(rows);
        for (size_type i = 0; i < rows; ++i)
            mtx[i].resize(cols);
        for (size_type i = 0; i < rows; ++i)
        {
            for (size_type j = 0; j < cols; ++j)
                mtx[i][j] = {};
        }
    }

    template <typename T>
    matrix<T>::~matrix(void) {}

    template <typename T>
    matrix<T>::matrix(matrix const &rhs)
    {
        size_type const rows = rhs.Rows(), cols = rhs.Cols();
        mtx.resize(rows);
        for (size_type i = 0; i < rows; ++i)
            mtx[i].resize(cols);
        for (size_type i = 0; i < rows; ++i)
        {
            for (size_type j = 0; j < cols; ++j)
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
    std::vector<T> &matrix<T>::operator[](size_type index)
    {
        return const_cast<std::vector<value_type>&>(static_cast<matrix<value_type> const &>(*this)[index]);
    }

    template <typename T>
    std::vector<T> const &matrix<T>::operator[](size_type index) const
    {
        if(0 > index || Rows() <= index)
            throw IndexOutOfBounds(index);
        return mtx[index];
    }

    template <typename T>
    bool matrix<T>::operator==(matrix const &rhs) const
    {
        size_type const rows = Rows(), cols = Cols();
        for (size_type i = 0; i < rows; ++i)
        {
            for (size_type j = 0; j < cols; ++j)
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
        size_type const rows = Rows(), cols = Cols();
        if (rows != rhs.Rows() || cols != rhs.Cols())
            throw IncompatibleMatrices("Addition", rows, cols, rhs.Rows(), rhs.Cols());
        for (size_type i = 0; i < rows; ++i)
        {
            for (size_type j = 0; j < cols; ++j)
                mtx[i][j] += rhs.mtx[i][j];
        }
        return *this;
    }

    template <typename T>
    matrix<T> &matrix<T>::operator-=(matrix const &rhs)
    {
        size_type const rows = Rows(), cols = Cols();
        if(rows != rhs.Rows() || cols != rhs.Cols())
            throw IncompatibleMatrices("Subtraction", rows, cols, rhs.Rows(), rhs.Cols());
        for (size_type i = 0; i < rows; ++i)
        {
            for (size_type j = 0; j < cols; ++j)
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
        size_type const rows = Rows(), cols = Cols();
        for (size_type i = 0; i < rows; ++i)
        {
            for (size_type j = 0; j < cols; ++j)
                mtx[i][j] *= rhs;
        }
        return *this;
    }

    template <typename T>
    typename matrix<T>::size_type matrix<T>::Rows(void) const
    {
        return mtx.size();
    }

    template <typename T>
    typename matrix<T>::size_type matrix<T>::Cols(void) const
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
        typename matrix<T>::size_type const l_rows = lhs.Rows(), l_cols = lhs.Cols(), r_rows = rhs.Rows(), r_cols = rhs.Cols();
        if(l_cols != r_rows)
            throw IncompatibleMatrices("Multiplication", l_rows, l_cols, r_rows, r_cols);
        typename matrix<T>::size_type const final_size = l_rows * r_cols;
        typename matrix<T>::size_type curr_size = 0;
        typename matrix<T>::size_type k = 0, l = 0;
        matrix<T> tmp(l_rows, r_cols);

        while (curr_size < final_size)
        {
            for (typename matrix<T>::size_type i = 0; i < r_cols; ++i)
            {
                typename matrix<T>::value_type sum = {};
                for (typename matrix<T>::size_type j = 0; j < l_cols; ++j)
                    sum += lhs[l][j] * rhs[j][k];
                tmp[l][k++] = sum, ++curr_size;
            }
            // condition for me to know that im done with the current row
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
        typename matrix<T>::size_type const rows = rhs.Rows(), cols = rhs.Cols();
        for (typename matrix<T>::size_type i = 0; i < rows; ++i)
        {
            for (typename matrix<T>::size_type j = 0; j < cols; ++j)
                os << rhs[i][j] << (j + 1 == cols ? "" : " ");
            os << std::endl;
        }
        return os;
    }
}
#endif