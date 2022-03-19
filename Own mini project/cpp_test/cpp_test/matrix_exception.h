#include <exception>
#include <cstdio>
#include <sstream>

class InvalidDimension : public std::exception
{
public:
    InvalidDimension(long long rows, long long cols, const char* s = nullptr);
    ~InvalidDimension(void) = default;

    virtual const char* what(void) const noexcept;

private:
    std::string msg;
};

class IndexOutOfBounds : public std::exception
{
public:
    IndexOutOfBounds(long long row, long long R, long long col, long long C);
    ~IndexOutOfBounds(void) = default;

    virtual const char* what(void) const noexcept;

private:
    std::string msg;
};

class IncompatibleMatrices : public std::exception
{
public:
    IncompatibleMatrices(const char* operation, long long l_rows, long long l_cols, long long r_rows, long long r_cols);
    ~IncompatibleMatrices(void) = default;

    virtual const char* what(void) const noexcept;

private:
    std::string msg;
};