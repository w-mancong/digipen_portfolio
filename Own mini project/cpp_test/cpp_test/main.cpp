#include "matrix.hpp"
#include <iostream>
#include <iomanip>

int main()
{
    const size_t SIZE = 5;
    ManCong::matrix b(3, 2);
    ManCong::matrix c(2, 2);

    b(0, 0) = 7.0f; b(0, 1) = 3.0f;
    b(1, 0) = 2.0f; b(1, 1) = 4.0f;
    b(2, 0) = 6.0f; b(2, 1) = 3.0f;
    std::cout << b << std::endl;

    b.Transpose();
    std::cout << b << std::endl;

    b.Transpose();
    std::cout << b << std::endl;
}