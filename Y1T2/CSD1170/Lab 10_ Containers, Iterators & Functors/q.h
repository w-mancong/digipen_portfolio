/*!*****************************************************************************
\file q.h
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 10
\date 25-03-2022
\brief
This file contains functions to find prime number and to calculate results of
exponent number
*******************************************************************************/
#ifndef Q_H
#define Q_H

#include <map>
#include <vector>
#include <algorithm>

namespace hlp2
{
    /*!*********************************************************************************
    \brief
        Descend class with the purpose of a functor to sort number on a descending order
    ***********************************************************************************/
    class Descend
    {
    public:
        /*!*********************************************************************************
        \brief
            Overloaded function operator to perform function behavior of sorting in a 
            descending order
        \param [in] lhs:
            Value on the left hand side
        \param [in] rhs:
            Value on the right hand side
        \return
            True if rhs is smaller than lhs
        ***********************************************************************************/
        template <typename T>
        bool operator()(T const &lhs, T const &rhs);
    };
    
    /*!*********************************************************************************
    \brief
        Overloaded function operator to perform function behavior of sorting in a
        descending order
    \param [in] lhs:
        Value on the left hand side
    \param [in] rhs:
        Value on the right hand side
    \return
        True if rhs is smaller than lhs
    ***********************************************************************************/
    template <typename T>
    bool Descend::operator()(T const& lhs, T const& rhs)
    {
        return rhs < lhs;
    }

    /*!*********************************************************************************
    \brief
        Takes in a half open-range array and check if the numbers are prime
    \param [in] beg:
        Pointer to the first element of the array
    \param [in] end:
        Pointer to the last element of the array
    \return
        A vector containing all the prime number in the half open-range array
    ***********************************************************************************/
    template <typename It>
    std::vector<unsigned> prime(It beg, It end)
    {
        std::vector<unsigned> res;
        for (auto it = beg; it != end; ++it)
        {
            size_t n = *it >> 1, flag = 0;
            for (size_t i = 2; i <= n; ++i)
            {
                if (*it % i)
                    continue;
                flag = 1;
                break;
            }
            if (flag || *it == 1 || !*it)
                continue;
            res.push_back(*it);
        }

        std::sort(res.begin(), res.end(), Descend());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
    }

    /*!*********************************************************************************
    \brief
        Finding the exponent of all the elements in the array
    \param [in] ar:
        Array containing all the elements to find the value to the power of ex
    \param [in] ex:
        Number of times the element will multiply by itself
    \return
        A map containing all the exponents of the results
    ***********************************************************************************/
    template <typename T, size_t N>
    std::map<T, T> pow(std::array<T, N> const &ar, int ex)
    {
        std::map<T, T> res;
        for (auto it = ar.begin(); it != ar.end(); ++it)
        {
            res[*it] = *it;
            for (int i = 0; i < ex - 1; ++i)
                res[*it] *= *it;
            if (!ex)
                res[*it] = static_cast<T>(1);
        }
        return res;
    }
}

#endif