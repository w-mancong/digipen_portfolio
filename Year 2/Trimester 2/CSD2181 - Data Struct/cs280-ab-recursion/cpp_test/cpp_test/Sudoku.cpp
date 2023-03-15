/*!
file:	Sudoku.cpp
author:	Wong Man Cong
email:	w.mancong\@digipen.edu
brief:	This file contains function definition for solving a sudoku

		All content © 2022 DigiPen Institute of Technology Singapore. All rights reserved.
*//*__________________________________________________________________________________*/
#include "Sudoku.h"

Sudoku::Sudoku(int basesize, SymbolType stype, SUDOKU_CALLBACK callback) : 
	m_Callback{ callback }, 
	m_BoardLen{ static_cast<size_t>(basesize) * static_cast<size_t>(basesize) },
	m_SymbolType{ stype }
{
	m_Stats.basesize = basesize;
	m_Board			 = new char[m_BoardLen * m_BoardLen];
}

Sudoku::~Sudoku()
{
	delete[] m_Board;
}

void Sudoku::SetupBoard(char const* values, size_t size)
{
	for (size_t i{}; i < size; ++i)
		*(m_Board + i) = *(values + i) == '.' ? EMPTY_CHAR : *(values + i);
}

void Sudoku::Solve()
{
	m_Callback(*this, m_Board, MessageType::MSG_STARTING, m_Stats.moves, m_Stats.basesize, -1, 0);

	MessageType const msg = PlaceValue(0, 0) ? MessageType::MSG_FINISHED_OK : MessageType::MSG_FINISHED_FAIL;
	m_Callback(*this, m_Board, msg, m_Stats.moves, m_Stats.basesize, -1, 0);
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

	size_t const index = GetIndex(x, y);
	unsigned const uIdx = static_cast<unsigned>(index);
	char val = m_SymbolType == SymbolType::SYM_NUMBER ? '1' : 'A';

	auto Place = [x, y, this](void)
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

	// if current index is already occupied, skip and place to the next cell
	if (*(m_Board + index) != EMPTY_CHAR)
		return Place();

	for (size_t i{}; i < m_BoardLen; ++i)
	{
		if ( m_Callback(*this, m_Board, MessageType::MSG_ABORT_CHECK, m_Stats.moves, m_Stats.basesize, uIdx, val) )
			return false;

		m_Callback(*this, m_Board, MessageType::MSG_PLACING, m_Stats.moves, m_Stats.basesize, uIdx, val);
		*(m_Board + index) = val;
		++m_Stats.moves, ++m_Stats.placed;

		// Check the current index is valid
		if (IsValid(x, y, val))
		{	// Place item in the next avaliable cell
			if (Place())
				return true;

			++m_Stats.backtracks;
			m_Callback(*this, m_Board, MessageType::MSG_REMOVING, m_Stats.moves, m_Stats.basesize, uIdx, val);
		}

		--m_Stats.placed, ++val;
		*(m_Board + index) = EMPTY_CHAR;
		m_Callback(*this, m_Board, MessageType::MSG_REMOVING, m_Stats.moves, m_Stats.basesize, uIdx, val);
	}

	return false;
}

bool Sudoku::IsValid(size_t x, size_t y, char val) const
{
	size_t const index = GetIndex(x, y);
	{	// Check if m_BoardLen x m_BoardLen grid contains the same value
		size_t const min_x = (x / m_Stats.basesize) * m_Stats.basesize,
					 min_y = (y / m_Stats.basesize) * m_Stats.basesize,
					 max_x = (x / m_Stats.basesize + 1ULL) * m_Stats.basesize,
					 max_y = (y / m_Stats.basesize + 1ULL) * m_Stats.basesize;

		for (size_t j{ min_y }; j < max_y; ++j)
		{
			for (size_t i{ min_x }; i < max_x; ++i)
			{
				size_t const currIdx = GetIndex(i, j);
				if (*(m_Board + currIdx) == val && currIdx != index)
					return false;
			}
		}
	}

	{	// Check if vertical/horizontal contains the same value
		for (size_t i{}; i < m_BoardLen; ++i)
		{
			size_t const rowIdx = GetIndex(i, y),
						 colIdx = GetIndex(x, i);

			if (index != rowIdx && *(m_Board + rowIdx) == val)
				return false;

			if (index != colIdx && *(m_Board + colIdx) == val)
				return false;
		}
	}
	
	return true;
}
