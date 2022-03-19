#include "matrix.hpp"
#include <iostream>
#include <iomanip>

int main()
{
    const size_t SIZE = 5;
    ManCong::matrix b(3, 2);
    ManCong::matrix c(2, 2);

    try 
    {
        b(3, 2);
    }
    catch (std::exception const& e)
    {
        std::cout << e.what() << std::endl;
    }
}