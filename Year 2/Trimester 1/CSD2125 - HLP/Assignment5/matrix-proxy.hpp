/*!*****************************************************************************
\file matrix-proxy.hpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Assignment 5
\date 17-10-2022
\brief
This file contains function definitions that uses a proxy class to allow clients
to access member variables of a dynamically allocated matrix with the [] operator
*******************************************************************************/
#ifndef MATRIX_PROXY_HPP
#define MATRIX_PROXY_HPP

namespace HLP3
{
    template <typename T>
    class Matrix
    {
    public:
        using value_type = T;
        using reference = value_type &;
        using const_reference = value_type const &;
        using pointer = value_type *;
        using const_pointer = value_type const *;
        using size_type = size_t;

        class Proxy
        {
        public:
            /*!*****************************************************************************
                \brief Constructor
            *******************************************************************************/
            Proxy(void) = default;

            /*!*****************************************************************************
                \brief Destructor
            *******************************************************************************/
            ~Proxy(void) noexcept;

            /*!*****************************************************************************
                \brief Dynamically allocate a 1D array of value_type

                \param cols [in]: Number of columns
            *******************************************************************************/
            void Allocate(size_type cols);

            /*!*****************************************************************************
                \brief Return a reference to value_type

                \param col [in]: Column index

                \return Reference to value_type
            *******************************************************************************/
            reference operator[](size_type col);

            /*!*****************************************************************************
                \brief Return a constant reference to value_type

                \param col [in]: Column index

                \return Reference to a constant value_type
            *******************************************************************************/
            const_reference operator[](size_type col) const;

        private:
            pointer data;
        };

        /*!*****************************************************************************
            \brief Constructor for matrix

            \param nr [in]: Number of rows
            \param nc [in]: Number of columns
        *******************************************************************************/
        Matrix(size_type nr, size_type nc);

        /*!*****************************************************************************
            \brief Copy constructor
        *******************************************************************************/
        Matrix(Matrix const &rhs);

        /*!*****************************************************************************
            \brief Move constructor
        *******************************************************************************/
        Matrix(Matrix &&rhs) noexcept;

        /*!*****************************************************************************
            \brief Constructor with initializer_list

            \param rhs [in]: List of value_type
        *******************************************************************************/
        Matrix(std::initializer_list<std::initializer_list<value_type>> rhs);

        /*!*****************************************************************************
            \brief Destructor
        *******************************************************************************/
        ~Matrix() noexcept;

        /*!*****************************************************************************
            \brief Assignment operator for both copy and move assignment

            \param rhs [in]: Matrix to be copy/moved

            \return The result of copying/moving rhs
        *******************************************************************************/
        Matrix &operator=(Matrix rhs);

        /*!*****************************************************************************
            \brief Get the number of rows in this matrix

            \return Total number of rows
        *******************************************************************************/
        size_type get_rows() const noexcept;

        /*!*****************************************************************************
            \brief Get the number of columns in this matrix

            \return Total number of columns
        *******************************************************************************/
        size_type get_cols() const noexcept;

        /*!*****************************************************************************
            \brief Return a reference to Proxy

            \param row [in]: Row index

            \return Reference to proxy
        *******************************************************************************/
        Proxy &operator[](size_type row);

        /*!*****************************************************************************
            \brief Return a const reference to Proxy

            \param row [in]: Row index

            \return A reference to a constant Proxy
        *******************************************************************************/
        Proxy const &operator[](size_type row) const;

    private:
        /*!*****************************************************************************
            \brief Swap the values of rhs with *this matrix

            \param rhs [in]: Values to be swapped with
        *******************************************************************************/
        void swap(Matrix &rhs);

        size_type rows{}, cols{};
        Proxy *data;
    };

    /*!*****************************************************************************
        \brief Addition of two matrix

        \param lhs [in]: Matrix on the left hand side of the operator
        \param rhs [in]: Matrix on the right hand side of the operator

        \return Result of the addition of the two matrix
    *******************************************************************************/
    template <typename T>
    Matrix<T> operator+(Matrix<T> const &lhs, Matrix<T> const &rhs);

    /*!*****************************************************************************
        \brief Subtraction of two matrix

        \param lhs [in]: Matrix on the left hand side of the operator
        \param rhs [in]: Matrix on the right hand side of the operator

        \return Result of the subtraction of the two matrix
    *******************************************************************************/
    template <typename T>
    Matrix<T> operator-(Matrix<T> const &lhs, Matrix<T> const &rhs);

    /*!*****************************************************************************
        \brief Multiplication of two matrix

        \param lhs [in]: Matrix on the left hand side of the operator
        \param rhs [in]: Matrix on the right hand side of the operator

        \return Result of the multiplication of the two matrix
    *******************************************************************************/
    template <typename T>
    Matrix<T> operator*(Matrix<T> const &lhs, Matrix<T> const &rhs);

    /*!*****************************************************************************
        \brief Multiplication of a matrix with a scalr

        \param lhs [in]: Matrix on the left hand side of the operator
        \param rhs [in]: Scalar value on the right hand side of the operator

        \return Result of the multiplication of the matrix with the scale value
    *******************************************************************************/
    template <typename T>
    Matrix<T> operator*(Matrix<T> const &lhs, T const &rhs);

    /*!*****************************************************************************
        \brief Multiplication of a matrix with a scalr

        \param lhs [in]: Scalar value on the left hand side of the operator
        \param rhs [in]: Matrix on the right hand side of the operator

        \return Result of the multiplication of the matrix with the scale value
    *******************************************************************************/
    template <typename T>
    Matrix<T> operator*(T const &lhs, Matrix<T> const &rhs);

    /*!*****************************************************************************
        \brief Check if two matrices are the same

        \param lhs [in]: Matrix on the left hand side of the operator
        \param rhs [in]: Matrix on the right hand side of the operator

        \return true if both matrices are the same, else false
    *******************************************************************************/
    template <typename T>
    bool operator==(Matrix<T> const &lhs, Matrix<T> const &rhs);

    /*!*****************************************************************************
        \brief Check if two matrices are not the same

        \param lhs [in]: Matrix on the left hand side of the operator
        \param rhs [in]: Matrix on the right hand side of the operator

        \return true if both matrices are not the same, else false
    *******************************************************************************/
    template <typename T>
    bool operator!=(Matrix<T> const &lhs, Matrix<T> const &rhs);

    template <typename T>
    Matrix<T>::Proxy::~Proxy(void) noexcept
    {
        if (data)
            delete[] data;
    }

    template <typename T>
    void Matrix<T>::Proxy::Allocate(size_type cols)
    {
        data = new value_type[cols]{};
    }

    template <typename T>
    typename Matrix<T>::reference Matrix<T>::Proxy::operator[](size_type col)
    {
        return const_cast<reference>(const_cast<Proxy const &>((*this))[col]);
    }

    template <typename T>
    typename Matrix<T>::const_reference Matrix<T>::Proxy::operator[](size_type col) const
    {
        return *(data + col);
    }

    template <typename T>
    Matrix<T>::Matrix(Matrix<T>::size_type nr, Matrix<T>::size_type nc) : rows{nr}, cols{nc}, data{nullptr}
    {
        data = new Proxy[nr]{};
        for (size_type i = 0; i < nr; ++i)
            (data + i)->Allocate(nc);
    }

    template <typename T>
    Matrix<T>::Matrix(Matrix const &rhs) : rows{rhs.rows}, cols{rhs.cols}, data{nullptr}
    {
        Matrix tmp(rows, cols);
        for (size_type i{}; i < rows; ++i)
        {
            for (size_type j{}; j < cols; ++j)
                tmp[i][j] = rhs[i][j];
        }
        swap(tmp);
    }

    template <typename T>
    Matrix<T>::Matrix(Matrix &&rhs) noexcept : rows{rhs.rows}, cols{rhs.cols}, data{nullptr}
    {
        swap(rhs);
    }

    template <typename T>
    Matrix<T>::Matrix(std::initializer_list<std::initializer_list<value_type>> rhs) : data{nullptr}
    {
        size_type const row{rhs.size()}, col{rhs.begin()->size()};
        // To check if all rows are of equal sizes
        for (auto const &v : rhs)
        {
            if (col != v.size())
                throw std::runtime_error{"bad initializer list"};
        }
        Matrix tmp{row, col};

        size_type i{}, j{};
        for (auto const &outer : rhs)
        {
            j = 0;
            for (auto const &inner : outer)
                tmp[i][j++] = inner;
            ++i;
        }

        swap(tmp);
    }

    template <typename T>
    Matrix<T>::~Matrix() noexcept
    {
        if (data)
            delete[] data;
    }

    template <typename T>
    Matrix<T> &Matrix<T>::operator=(Matrix rhs)
    {
        swap(rhs);
        return *this;
    }

    template <typename T>
    typename Matrix<T>::size_type Matrix<T>::get_rows() const noexcept
    {
        return rows;
    }

    template <typename T>
    typename Matrix<T>::size_type Matrix<T>::get_cols() const noexcept
    {
        return cols;
    }

    template <typename T>
    typename Matrix<T>::Proxy &Matrix<T>::operator[](size_type row)
    {
        return const_cast<Proxy &>(const_cast<Matrix const &>((*this))[row]);
    }

    template <typename T>
    typename Matrix<T>::Proxy const &Matrix<T>::operator[](size_type row) const
    {
        return *(data + row);
    }

    template <typename T>
    void Matrix<T>::swap(Matrix &rhs)
    {
        std::swap(data, rhs.data);
        std::swap(rows, rhs.rows);
        std::swap(cols, rhs.cols);
    }

    template <typename T>
    Matrix<T> operator+(Matrix<T> const &lhs, Matrix<T> const &rhs)
    {
        using size_type = typename Matrix<T>::size_type;
        if (lhs.get_rows() != rhs.get_rows() || lhs.get_cols() != rhs.get_cols())
            throw std::runtime_error{"operands for matrix addition must have same dimensions"};
        size_type const rows = lhs.get_rows(), cols = lhs.get_cols();
        Matrix<T> res{lhs};
        for (size_type i{}; i < rows; ++i)
        {
            for (size_type j{}; j < cols; ++j)
                res[i][j] += rhs[i][j];
        }
        return res;
    }

    template <typename T>
    Matrix<T> operator-(Matrix<T> const &lhs, Matrix<T> const &rhs)
    {
        using size_type = typename Matrix<T>::size_type;
        if (lhs.get_rows() != rhs.get_rows() || lhs.get_cols() != rhs.get_cols())
            throw std::runtime_error{"operands for matrix subtraction must have same dimensions"};
        size_type const rows = lhs.get_rows(), cols = lhs.get_cols();
        Matrix<T> res{lhs};
        for (size_type i{}; i < rows; ++i)
        {
            for (size_type j{}; j < cols; ++j)
                res[i][j] -= rhs[i][j];
        }
        return res;
    }

    template <typename T>
    Matrix<T> operator*(Matrix<T> const &lhs, Matrix<T> const &rhs)
    {
        using value_type = typename Matrix<T>::value_type;
        using size_type = typename Matrix<T>::size_type;
        const size_type l_rows = lhs.get_rows(), l_cols = lhs.get_cols(), r_rows = rhs.get_rows(), r_cols = rhs.get_cols();
        if (l_cols != r_rows)
            throw std::runtime_error{"number of columns in left operand must match number of rows in right operand"};
        Matrix<T> res(l_rows, r_cols);
        for (size_type i = 0; i < l_rows; ++i)
        {
            for (size_type j = 0; j < r_cols; ++j)
            {
                res[i][j] = static_cast<value_type>(0);
                for (size_type k = 0; k < r_rows; ++k)
                    res[i][j] += lhs[i][k] * rhs[k][j];
            }
        }
        return res;
    }

    template <typename T>
    Matrix<T> operator*(Matrix<T> const &lhs, T const &rhs)
    {
        using size_type = typename Matrix<T>::size_type;
        size_type const rows = lhs.get_rows(), cols = lhs.get_cols();
        Matrix<T> res{lhs};
        for (size_type i{}; i < rows; ++i)
        {
            for (size_type j{}; j < cols; ++j)
                res[i][j] *= rhs;
        }
        return res;
    }

    template <typename T>
    Matrix<T> operator*(T const &lhs, Matrix<T> const &rhs)
    {
        return rhs * lhs;
    }

    template <typename T>
    bool operator==(Matrix<T> const &lhs, Matrix<T> const &rhs)
    {
        using size_type = typename Matrix<T>::size_type;
        if (lhs.get_rows() != rhs.get_rows() || lhs.get_cols() != rhs.get_cols())
            return false;

        size_type const rows = lhs.get_rows(), cols = lhs.get_cols();

        for (size_type i{}; i < rows; ++i)
        {
            for (size_type j{}; j < cols; ++j)
            {
                if (lhs[i][j] != rhs[i][j])
                    return false;
            }
        }
        return true;
    }

    template <typename T>
    bool operator!=(Matrix<T> const &lhs, Matrix<T> const &rhs)
    {
        return !(lhs == rhs);
    }
}

#endif
