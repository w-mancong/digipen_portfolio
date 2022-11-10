template <uint64_t Ns, uint64_t... tail>
struct make_sequence_impl
{
  using type = typename make_sequence_impl<Ns - 1, tail..., 1ULL << Ns>::type;
};

template <uint64_t... tail>
struct make_sequence_impl<0, tail...>
{
  using type = index_sequence<tail..., 1ULL << 0>;
};

template <uint64_t Ns>
using make_sequence = typename make_sequence_impl<Ns>::type;