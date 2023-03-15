/*!
file:	Sudoku.h
author:	Wong Man Cong
email:	w.mancong\@digipen.edu
brief:	This file contains function declarations for solving a sudoku

		All content © 2022 DigiPen Institute of Technology Singapore. All rights reserved.
*//*__________________________________________________________________________________*/
//---------------------------------------------------------------------------
#ifndef SUDOKUH
#define SUDOKUH
//---------------------------------------------------------------------------
#include <cstddef> /* size_t */

//! The Sudoku class
class Sudoku
{
public:
	//! Used by the callback function 
	enum MessageType
	{
		MSG_STARTING,      //!< the board is setup, ready to go
		MSG_FINISHED_OK,   //!< finished and found a solution
		MSG_FINISHED_FAIL, //!< finished but no solution found
		MSG_ABORT_CHECK,   //!< checking to see if algorithm should continue
		MSG_PLACING,       //!< placing a symbol on the board
		MSG_REMOVING       //!< removing a symbol (back-tracking)
	};

	//! 1-9 for 9x9, A-P for 16x16, A-Y for 25x25
	enum SymbolType { SYM_NUMBER, SYM_LETTER };

	//! Represents an empty cell (the driver will use a . instead)
	constexpr static char EMPTY_CHAR = ' ';

	//! Implemented in the client and called during the search for a solution
	using SUDOKU_CALLBACK = bool(*) (Sudoku const& sudoku, // the gameboard object itself
									 char const* board,    // one-dimensional array of symbols
									 MessageType message,  // type of message
									 size_t move,          // the move number
									 unsigned basesize,    // 3, 4, 5, etc. (for 9x9, 16x16, 25x25, etc.)
									 unsigned index,       // index (0-based) of current cell
									 char value            // symbol (value) in current cell
									 );			

	//! Statistics as the algorithm works
	struct SudokuStats
	{
		int basesize;      //!< 3, 4, 5, etc.
		int placed;        //!< the number of valid values the algorithm has placed
		size_t moves;      //!< total number of values that have been tried
		size_t backtracks; //!< total number of times the algorithm backtracked

		//!< Default constructor
		SudokuStats() : basesize(0), placed(0), moves(0), backtracks(0) {}
	};

	/*!*********************************************************************************
		\brief Constructor for Sudoku

		\param [in] basesize: Determines if the board is 3x3, 4x4 or 5x5
		\param [in] stype: Type of character used for printing the board
		\param [in] callback: Callback function provided by client
	***********************************************************************************/
	Sudoku(int basesize, SymbolType stype = SYM_NUMBER, SUDOKU_CALLBACK callback = nullptr);

	/*!*********************************************************************************
		\brief Destructor for Sudoku
	***********************************************************************************/
	~Sudoku();

	/*!*********************************************************************************
		\brief Initializes the sudoku board

		\param [in] values: Pointer containing the values of the board
		\param [in] size: Size of the entire board
	***********************************************************************************/
	void SetupBoard(char const* values, size_t size);

	/*!*********************************************************************************
		\brief Calling the recursive function to find a solution to the sudoku puzzle
	***********************************************************************************/
	void Solve();

	/*!*********************************************************************************
		\brief Accessor function to retrieve the board data
	***********************************************************************************/
	const char* GetBoard() const;

	/*!*********************************************************************************
		\brief Accessor function to retrieve the stats of sudoku data
	***********************************************************************************/
	SudokuStats GetStats() const;

private:
	/*!*********************************************************************************
		\brief Helper function to calculate the index of a 1D dynamic array

		\param [in] x, y: X/Y coordinate of the board
	***********************************************************************************/
	size_t GetIndex(size_t x, size_t y) const;

	/*!*********************************************************************************
		\brief Recursive function to place a valid value onto the board

		\param [in] x, y: X/Y coordinate of the board

		\return true if value is placed on board successfull, else false
	***********************************************************************************/
	bool PlaceValue(size_t x, size_t y);

	/*!*********************************************************************************
		\brief Check if the position at x and y is a valid placement on the sudoku board

		\param [in] x, y: X/Y coordinate of the board
		\param [in] val: Value to be checked with on the board

		\return true if x and y is a valid placement for val, else false
	***********************************************************************************/
	bool IsValid(size_t x, size_t y, char val) const;

	SudokuStats m_Stats{};
	SUDOKU_CALLBACK const m_Callback{ nullptr }; // Callback function provided by the client
	char* m_Board{ nullptr };					 // Pointer to store the sudoku board
	size_t const m_BoardLen{};					 // Length of the board
	SymbolType const m_SymbolType{ SYM_NUMBER };
};

#endif  // SUDOKUH
