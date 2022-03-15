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
    ManCong::matrix<3, 2> b;
    ManCong::matrix<2, 2> c;

    b[0][0] = 7.0f; b[0][1] = 3.0f;
    b[1][0] = 2.0f; b[1][1] = 4.0f;

    c[0][0] = 8.0f; c[0][1] = 3.0f;
    c[1][0] = 4.0f; c[1][1] = 5.0f;

    b *= c;

    ManCong::matrix<3, 2> a = b * c;
}