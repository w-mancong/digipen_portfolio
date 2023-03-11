#include "Sudoku.h"
#include <iostream>

Sudoku::Sudoku(int basesize, SymbolType stype, SUDOKU_CALLBACK callback)
{
	m_Callback = callback;
	m_BoardLen = static_cast<size_t>(basesize) * static_cast<size_t>(basesize);
	m_Stats.basesize = basesize;
}

Sudoku::~Sudoku()
{
	delete[] m_Board;
	delete[] m_OriginalCell;
}

void Sudoku::SetupBoard(char const* values, int size)
{
	m_Size = static_cast<size_t>(size);
	m_Board = new char[m_Size + 1] {};
	m_OriginalCell = new bool[m_Size] {};
	*(m_Board + m_Size) = '\0';

	size_t i{};
	char const* it = values; char* ptr = m_Board;
	bool* bptr = m_OriginalCell;
	do
	{
		if ( *(it + i) == '.')
			 *(ptr + i) = ' ';
		else
		{
			*(ptr + i) = *(it + i);
			*(bptr + i) = true;
		}
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

size_t Sudoku::GetIndex(size_t x, size_t y)
{
	return x * m_BoardLen + y;
}
