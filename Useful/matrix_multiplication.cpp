#include <iostream>

int main(void)
{
    size_t constexpr MAX{ 3 };
    for (size_t i = 0; i < MAX; ++i)
    {
        for (size_t j = 0; j < MAX; ++j)
        {
            std::cout << "res(" << i << ", " << j << ") = ";
            for (size_t k = 0; k < MAX; ++k)
                std::cout << "lhs(" << i << ", " << k << ") * rhs(" << k << ", " << j << ")" << (k + 1 >= MAX ? ";" : " + ");
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}