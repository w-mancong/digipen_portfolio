#include "matrix.hpp"
#include <iostream>
#include <iomanip>

int main()
{
    const size_t SIZE = 5;
    ManCong::matrix m1(3, 3), m2(3, 3);

    m1(0, 0) = 1.0f; m1(0, 1) = 6.0f; m1(0, 2) = 2.0f;
    m1(1, 0) = 4.0f; m1(1, 1) = 8.0f; m1(1, 2) = 7.0f;
    m1(2, 0) = 9.0f; m1(2, 1) = 5.0f; m1(2, 2) = 3.0f;

    m2(0, 0) = 8.0f; m2(0, 1) = 4.0f; m2(0, 2) = 3.0f;
    m2(1, 0) = 6.0f; m2(1, 1) = 5.0f; m2(1, 2) = 1.0f;
    m2(2, 0) = 2.0f; m2(2, 1) = 7.0f; m2(2, 2) = 9.0f;

    std::cout << m1 * m2 << std::endl << std::endl;

    m1.Transpose(); m2.Transpose();

    std::cout << (m2 * m1).Transpose() << std::endl;
}