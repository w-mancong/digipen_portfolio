template <uint64_t Ns, uint64_t... tail>
constexpr auto make_sequence_impl()
{
    if constexpr (Ns == 0)
        return index_sequence<tail..., 1>{};
    constexpr make_sequence_impl<Ns - 1, tail..., 1ULL << Ns>();
}

template <uint64_t Ns>
using make_sequence = decltype(make_sequence_impl<Ns>());