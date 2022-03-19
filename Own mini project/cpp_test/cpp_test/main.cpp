#include "matrix.hpp"
#include <iostream>
#include <iomanip>

int main()
{
    const size_t SIZE = 5;
    ManCong::matrix b(4, 4);
    ManCong::matrix c(2, 2);

    b(0, 0) = 2.0f; b(0, 1) = 3.0f; b(0, 2) = 6.0f; b(0, 3) = 1.0f;
    b(1, 0) = 9.0f; b(1, 1) = 7.0f; b(1, 2) = 3.0f; b(1, 3) = 5.0f;
    b(2, 0) = 5.0f; b(2, 1) = 5.0f; b(2, 2) = 2.0f; b(2, 3) = 8.0f;
    b(3, 0) = 5.0f; b(3, 1) = 6.0f; b(3, 2) = 2.0f; b(3, 3) = 3.0f;

    std::cout << b << std::endl;
    std::cout << b.Determinant() << std::endl;
}