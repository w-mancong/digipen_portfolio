#include "Sudoku.h"

Sudoku::Sudoku(int basesize, SymbolType stype, SUDOKU_CALLBACK callback) : m_Callback{ callback }, m_SymbolType{ stype }, m_BoardLen{ static_cast<size_t>(basesize) * static_cast<size_t>(basesize) }
{
	m_Stats.basesize = basesize;
	m_Board			 = new char[m_BoardLen * m_BoardLen];
}

Sudoku::~Sudoku()
{
	delete[] m_Board;
}

void Sudoku::SetupBoard(char const* values, int size)
{
	for (size_t i{}; i < static_cast<size_t>(size); ++i)
		*(m_Board + i) = *(values + i) == '.' ? ' ' : *(values + i);
}

void Sudoku::Solve()
{
	m_Callback(*this, m_Board, MessageType::MSG_STARTING, m_Stats.moves, m_Stats.basesize, -1, 0);

	if( PlaceValue(0, 0) )
		m_Callback(*this, m_Board, MessageType::MSG_FINISHED_OK, m_Stats.moves, m_Stats.basesize, -1, 0);
	else
		m_Callback(*this, m_Board, MessageType::MSG_FINISHED_FAIL, m_Stats.moves, m_Stats.basesize, -1, 0);
}

const char* Sudoku::GetBoard() const
{
	return m_Board;
}

Sudoku::SudokuStats Sudoku::GetStats() const
{
	return m_Stats;
}

size_t Sudoku::GetIndex(size_t x, size_t y) const
{
	return y * m_BoardLen + x;
}

bool Sudoku::PlaceValue(size_t x, size_t y)
{
	if ( m_BoardLen == y )
		return true;

	size_t index = GetIndex(x, y);
	char val = m_SymbolType == SymbolType::SYM_NUMBER ? '1' : 'A';

	auto Verification = [x, y, this](void)
	{
		if (m_BoardLen - 1 != x)
		{
			if (PlaceValue(x + 1, y))
				return true;
		}
		else
		{
			if (PlaceValue(0, y + 1))
				return true;
		}
		return false;
	};

	if (*(m_Board + index) != ' ')
		return Verification();

	for (size_t i{}; i < m_BoardLen; ++i)
	{
		if ( m_Callback(*this, m_Board, MessageType::MSG_ABORT_CHECK, m_Stats.moves, m_Stats.basesize, index, val) )
			return false;

		m_Callback(*this, m_Board, MessageType::MSG_PLACING, m_Stats.moves, m_Stats.basesize, index, val);
		*(m_Board + index) = val;
		++m_Stats.moves, ++m_Stats.placed;


	}

	return false;
}