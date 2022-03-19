#include "matrix_exception.h"

/******************************************************************************************************************
                                           Invalid Deimensions
******************************************************************************************************************/
InvalidDimension::InvalidDimension(int rows, int cols, const char* s) : msg{ "" }
{
    const char* ex = "Invalid Dimension Exception: ", * err = "is an invalid dimension for ";
    if (s)
        sprintf_s(msg, "%s%d and %d are invalid dimensions for %s", ex, rows, cols, s);
    else if (0 > rows && 0 > cols)
        sprintf_s(msg, "%s%d and %d are invalid dimensions for rows and columns respectively", ex, rows, cols);
    else if (0 > rows)
        sprintf_s(msg, "%s%d %srows", ex, rows, err);
    else if (0 > cols)
        sprintf_s(msg, "%s%d %scolumns", ex, cols, err);
}

const char* InvalidDimension::what(void) const throw()
{
    return msg;
}

/******************************************************************************************************************
                                            Index Out of Bound
******************************************************************************************************************/
IndexOutOfBounds::IndexOutOfBounds(int row, int R, int col, int C) : msg{ "" }
{
    if ((0 > row || R <= row) && (0 > col || C <= col))
        sprintf_s(msg, "Index Out Of Bounds Exception: %d and %d are invalid index for rows and columns respectively", row, col);
    else if (0 > row || R <= row)
        sprintf_s(msg, "Index Out Of Bounds Exception: %d is an invalid index for rows", row);
    else if (0 > col || C <= col)
        sprintf_s(msg, "Index Out Of Bounds Exception: %d is an invalid index for columns", col);
}

const char* IndexOutOfBounds::what(void) const throw()
{
    return msg;
}

/******************************************************************************************************************
                                           Incompatible Matrices
******************************************************************************************************************/
IncompatibleMatrices::IncompatibleMatrices(const char* operation, int l_rows, int l_cols, int r_rows, int r_cols) : msg{ "" }
{
    sprintf(msg, "Incompatible Matrices Exception: %s of LHS matrix with dimensions %d X %d and RHS matrix with dimensions %d X %d is undefined", operation, l_rows, l_cols, r_rows, r_cols);
}

const char* IncompatibleMatrices::what(void) const throw()
{
    return msg;
}