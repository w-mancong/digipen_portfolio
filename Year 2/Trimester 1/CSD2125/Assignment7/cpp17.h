/*!*****************************************************************************
\file cpp17.h
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Assignment 7
\date 10-11-2022
\brief
This file contains functions that evaluates 2^N at compile time
*******************************************************************************/

/*!*****************************************************************************
    \brief A constexpr function that will be evaulated at compile time to pack 
    2^N into index_sequence for printing

    \return Will be deduced by the compiler
*******************************************************************************/
template <uint64_t Ns, uint64_t... tail>
auto make_sequence_impl()
{
    if constexpr (Ns == 0)
        return index_sequence<tail..., 1>{};
    else
        return make_sequence_impl<Ns - 1, tail..., 1ULL << Ns>();
}

/*!*****************************************************************************
    \brief type definition for deducing the type for make_sequence
*******************************************************************************/
template <uint64_t Ns>
using make_sequence = decltype(make_sequence_impl<Ns>());