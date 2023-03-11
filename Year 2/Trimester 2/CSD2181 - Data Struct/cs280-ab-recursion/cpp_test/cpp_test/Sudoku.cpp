#include "Sudoku.h"

Sudoku::Sudoku(int basesize, SymbolType stype, SUDOKU_CALLBACK callback)
{
	m_Callback = callback;
	m_BoardLen = static_cast<size_t>(basesize) * static_cast<size_t>(basesize);
	m_Stats.basesize = basesize;
}

Sudoku::~Sudoku()
{
	delete[] m_Board;
}

void Sudoku::SetupBoard(char const* values, int size)
{
	size_t const SIZE = static_cast<size_t>(size);
	m_Board = new char[SIZE + 1];
	*(m_Board + SIZE) = '\0';

	size_t i{};
	char const* it = values; char* ptr = m_Board;
	do
	{
		if (*(it + i) == '.')
			*(ptr + i) = ' ';
		else
			*(ptr + i) = *(it + i);
	} while ( *(it + ++i) );
}

void Sudoku::Solve()
{

}

const char* Sudoku::GetBoard() const
{
	return m_Board;
}

Sudoku::SudokuStats Sudoku::GetStats() const
{
	return m_Stats;
}
