template <uint64_t Ns, bool odd, uint64_t... package>
struct make_sequence_impl
{
    using type = typename make_sequence_impl<Ns, odd, package...>::type;
};

template <uint64_t... package>
struct make_sequence_impl<1, true, package...>
{
    using type = index_sequence<package...>;
};

template <uint64_t Ns>
using make_sequence = typename make_sequence_impl<Ns, Ns % 2>::type;