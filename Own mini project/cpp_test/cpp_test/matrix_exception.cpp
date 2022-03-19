#include "matrix_exception.h"

/******************************************************************************************************************
                                           Invalid Deimensions
******************************************************************************************************************/
InvalidDimension::InvalidDimension(long long rows, long long cols, const char* s) : msg{ "" }
{
    const char* ex = "Invalid Dimension Exception: ", * err = "is an invalid dimension for ";
    std::ostringstream oss;
    if (s)
        oss << ex << rows << " and " << cols << " are invalid dimensions for " << s;
    else if (0 > rows && 0 > cols)
        oss << ex << rows << " and " << cols << " are invalid dimensions for rows and columns respectively";
    else if (0 > rows)
        oss << ex << rows << " " << err << "rows";
    else if (0 > cols)
        oss << ex << cols << " " << err << "columns";
    msg = oss.str();
}

const char* InvalidDimension::what(void) const noexcept
{
    return msg.c_str();
}

/******************************************************************************************************************
                                            Index Out of Bound
******************************************************************************************************************/
IndexOutOfBounds::IndexOutOfBounds(long long row, long long R, long long col, long long C) : msg{ "" }
{
    std::ostringstream oss;
    if ((0 > row || R <= row) && (0 > col || C <= col))
        oss << "Index Out Of Bounds Exception: " << row << " and " << col << " are invalid index for rows and columns respectively";
    else if (0 > row || R <= row)
        oss << "Index Out Of Bounds Exception: " << row << " is an invalid index for rows";
    else if (0 > col || C <= col)
        oss << "Index Out Of Bounds Exception: " << col << " is an invalid index for columns";
}

const char* IndexOutOfBounds::what(void) const noexcept
{
    return msg.c_str();
}

/******************************************************************************************************************
                                           Incompatible Matrices
******************************************************************************************************************/
IncompatibleMatrices::IncompatibleMatrices(const char* operation, long long l_rows, long long l_cols, long long r_rows, long long r_cols) : msg{ "" }
{
    std::ostringstream oss;
    oss << "Incompatible Matrices Exception: " << operation << "of LHS matrix with dimensions" << l_rows << " X " << l_cols << " and RHS matrix with dimensions " << r_rows << " X " << r_cols << " is undefined";
    msg = oss.str();
}

const char* IncompatibleMatrices::what(void) const noexcept
{
    return msg.c_str();
}