#include "matrix.hpp"
#include <iostream>
#include <iomanip>

struct A
{
    enum { Rows = 3, Cols = 4 };
    int matrix[Rows][Cols];
    int(&operator [](size_t i))[Cols] 
    {
        return matrix[i];
    }
    void print(void) const;
};

int main()
{
    const size_t SIZE = 5;
    ManCong::matrix b(3, 2);
    ManCong::matrix c(SIZE, SIZE);

    std::cout << c << std::endl;

    b(0, 0) = 7.0f; b(0, 1) = 3.0f;
    b(1, 0) = 2.0f; b(1, 1) = 4.0f;
    b(2, 0) = 6.0f; b(2, 1) = 3.0f;

    c(0, 0) = 8.0f; c(0, 1) = 3.0f;
    c(1, 0) = 4.0f; c(1, 1) = 5.0f;

    //b *= c;

    //std::cout << b(1, 1) << std::endl;

    //ManCong::matrix<3, 2> a = b * c;
    //a.Determinant();
    //a.Indentity();
    //std::cout << a << std::endl;
}