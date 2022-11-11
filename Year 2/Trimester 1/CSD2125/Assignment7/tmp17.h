// use template meta programming to solve 3x + 1

template <uint64_t Ns, uint64_t... package>
auto make_sequence_impl()
{
    if constexpr (Ns == 1)
        return index_sequence<package...>{};
    else if constexpr (Ns % 2)
        return make_sequence_impl<3 * Ns + 1, package..., 3 * Ns + 1>();
    else
        return make_sequence_impl<Ns / 2, package..., Ns / 2>();
}

template <uint64_t Ns>
using make_sequence = decltype(make_sequence_impl<Ns>());