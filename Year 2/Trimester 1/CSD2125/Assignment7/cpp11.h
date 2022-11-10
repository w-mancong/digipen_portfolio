/*!*****************************************************************************
\file cpp11.h
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Assignment 7
\date 10-11-2022
\brief
This file contains variadic function template that will evaluate 2^N at compile time
*******************************************************************************/

/*!*****************************************************************************
    \brief To pack 2^N into index_sequence
*******************************************************************************/
template <uint64_t Ns, uint64_t... tail>
struct make_sequence_impl
{
  using type = typename make_sequence_impl<Ns - 1, tail..., 1ULL << Ns>::type;
};

/*!*****************************************************************************
    \brief Partial specialisation template for when N reaches 0, and pass the
    packed parameters of tail into index_sequence
*******************************************************************************/
template <uint64_t... tail>
struct make_sequence_impl<0, tail...>
{
  using type = index_sequence<tail..., 1ULL << 0>;
};

template <uint64_t Ns>
using make_sequence = typename make_sequence_impl<Ns>::type;