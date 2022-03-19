#include <exception>
#include <cstdio>

namespace
{
    const size_t BUFFER = 1024;
}

class InvalidDimension : public std::exception
{
public:
    InvalidDimension(int rows, int cols, const char* s = nullptr);
    ~InvalidDimension(void) = default;

    virtual const char* what(void) const throw();

private:
    char msg[BUFFER];
};

class IndexOutOfBounds : public std::exception
{
public:
    IndexOutOfBounds(int row, int R, int col, int C);
    ~IndexOutOfBounds(void) = default;

    virtual const char* what(void) const throw();

private:
    char msg[BUFFER];
};

class IncompatibleMatrices : public std::exception
{
public:
    IncompatibleMatrices(const char* operation, int l_rows, int l_cols, int r_rows, int r_cols);
    ~IncompatibleMatrices(void) = default;

    virtual const char* what(void) const throw();

private:
    char msg[BUFFER];
};