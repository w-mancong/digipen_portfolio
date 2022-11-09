#include "index_sequence.h"

template <size_t N>
struct make_sequence
{
    enum : uint64_t { value = 1i64 << N };
    static void print(void)
    {
        index_sequence<value>::print();
    }
};