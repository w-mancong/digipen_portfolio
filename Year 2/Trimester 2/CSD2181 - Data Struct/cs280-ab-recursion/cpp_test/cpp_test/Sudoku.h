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
	const static char EMPTY_CHAR = ' ';

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

	// Constructor
	Sudoku(int basesize, SymbolType stype = SYM_NUMBER, SUDOKU_CALLBACK callback = nullptr);

	// Destructor
	~Sudoku();

	// The client (driver) passed the board in the values parameter
	void SetupBoard(char const* values, size_t size);

	// Once the board is setup, this will start the search for the solution
	void Solve();

	// For debugging with the driver
	const char* GetBoard() const;
	SudokuStats GetStats() const;

private:
	size_t GetIndex(size_t x, size_t y) const;
	bool PlaceValue(size_t x, size_t y);
	bool IsValid(size_t x, size_t y, char val) const;

	SudokuStats m_Stats{};
	SUDOKU_CALLBACK const m_Callback{ nullptr }; // Callback function provided by the client
	char* m_Board{ nullptr };					 // Pointer to store the sudoku board
	size_t const m_BoardLen{};					 // Length of the board
	SymbolType const m_SymbolType{ SYM_NUMBER };
};

#endif  // SUDOKUH
