#include <iostream>

template <long N>
struct make_sequence
{
    static void print(void)
    {
    }
};

template <>
struct make_sequence<0>
{
    enum
    {
        value = 1
    };
    static void print(void)
    {
        std::cout << value << std::endl;
    }
};