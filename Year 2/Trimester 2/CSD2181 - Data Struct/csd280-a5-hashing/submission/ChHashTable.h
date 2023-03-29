/*!
file:   ChHashTable.h
author:	Wong Man Cong
email:	w.mancong\@digipen.edu
brief:  This file contains function declaration for a templated Hash Table where the
		collision resolution is solved by chaining (linked-list)

		All content © 2023 DigiPen Institute of Technology Singapore. All rights reserved.
*/
/*__________________________________________________________________________________*/
//---------------------------------------------------------------------------
#ifndef CHHASHTABLEH
#define CHHASHTABLEH
//---------------------------------------------------------------------------

#include <string>
#include <cmath>
#include "ObjectAllocator.h"
#include "support.h"

// client-provided hash function: takes a key and table size,
// returns an index in the table.
using HASHFUNC = unsigned (*)(const char *, unsigned);

// Max length of our "string" keys
const unsigned MAX_KEYLEN = 10;

class HashTableException
{
private:
	int error_code_;
	std::string message_;

public:
	HashTableException(int ErrCode, const std::string &Message) : error_code_(ErrCode), message_(Message){};

	virtual ~HashTableException()
	{
	}

	virtual int code() const
	{
		return error_code_;
	}

	virtual const char *what() const
	{
		return message_.c_str();
	}
	enum HASHTABLE_EXCEPTION
	{
		E_ITEM_NOT_FOUND,
		E_DUPLICATE,
		E_NO_MEMORY
	};
};

// HashTable statistical info
struct HTStats
{
	HTStats(void) : Count_(0), TableSize_(0), Probes_(0), Expansions_(0),
					HashFunc_(nullptr), Allocator_(nullptr){};
	unsigned Count_;			 // Number of elements in the table
	unsigned TableSize_;		 // Size of the table (total slots)
	unsigned Probes_;			 // Number of probes performed
	unsigned Expansions_;		 // Number of times the table grew
	HASHFUNC HashFunc_;			 // Pointer to primary hash function
	ObjectAllocator *Allocator_; // The allocator in use (may be 0)
};

/*!*********************************************************************************
	\brief Declaration of a templated chained hash table
***********************************************************************************/
template <typename T>
class ChHashTable
{
public:
	using FREEPROC = void (*)(T);

	/*!*********************************************************************************
		\brief Struct containing the configuration for HashTable
	***********************************************************************************/
	struct HTConfig
	{
		HTConfig(unsigned InitialTableSize,
				 HASHFUNC HashFunc,
				 double MaxLoadFactor = 3.0,
				 double GrowthFactor = 2.0,
				 FREEPROC FreeProc = nullptr) :

												// The number of slots in the table initially.
												InitialTableSize_(InitialTableSize),
												// The hash function used in all cases.
												HashFunc_(HashFunc),
												// The maximum "fullness" of the table.
												MaxLoadFactor_(MaxLoadFactor),
												// The factor by which the table grows.
												GrowthFactor_(GrowthFactor),
												// The method provided by the client that may need to be called when
												// data in the table is removed.
												FreeProc_(FreeProc)
		{
		}

		unsigned InitialTableSize_{};
		HASHFUNC HashFunc_{};
		double MaxLoadFactor_{};
		double GrowthFactor_{};
		FREEPROC FreeProc_{};
	};

	/*!*********************************************************************************
		\brief Node structure that is used to contain all the node data
	***********************************************************************************/
	struct ChHTNode
	{
		char Key[MAX_KEYLEN]{}; // Key is a string
		T Data{};				// Client data
		ChHTNode *Next{nullptr};
		ChHTNode(T const &data) : Data(data){}; // constructor
	};

	using HashNode = ChHTNode *;

	// Each list has a special head pointer
	struct ChHTHeadNode
	{
		HashNode Nodes;
		ChHTHeadNode() : Nodes(0), Count(0){};
		int Count; // For testing
	};

	using HashHeadNode = ChHTHeadNode;
	using HashTable = HashHeadNode *;

	/*!*********************************************************************************
		\brief Constructor for ChHashTable

		\param [in] config: Configuration settings used for hash table
		\param [in] allocator: Memory allocator
	***********************************************************************************/
	ChHashTable(HTConfig const &config, ObjectAllocator *allocator = nullptr);

	/*!*********************************************************************************
		\brief Destructor
	***********************************************************************************/
	~ChHashTable();

	/*!*********************************************************************************
		\brief Insert a node into the hash table based on it's key

		\param [in] key: Used as an identifier for the associative array
		\param [in] data: The data stored along with the key

		\exception
		E_DUPLICATE is thrown if there is a duplicate key already existing
		E_NO_MEMORY is thrown if allocator fails to allocate memory
	***********************************************************************************/
	void insert(char const *key, T const &data);

	/*!*********************************************************************************
	\brief Remove a node based on it's key

	\param [in] key: Used as an identifier for the associative array

	\exception
		E_ITEM_NOT_FOUND is thrown if key does not exist
	***********************************************************************************/
	void remove(char const *key);

	/*!*********************************************************************************
		\brief Find and return the data by the key

		\param [in] key: Used as an identifier for the associative array

		\exception
		E_ITEM_NOT_FOUND is thrown if key does no exist
	***********************************************************************************/
	const T &find(char const *key) const;

	/*!*********************************************************************************
		\brief Remove all item from the hash table. Hash table will not be deallocated
	***********************************************************************************/
	void clear();

	/*!*********************************************************************************
		\brief Return the current stats of the HashTable
	***********************************************************************************/
	HTStats GetStats() const;

	/*!*********************************************************************************
		\brief Return a pointer to the table
	***********************************************************************************/
	ChHTHeadNode const *GetTable() const;

private:
	/*!*********************************************************************************
		\brief Helper function to retrieve the index based on the key

		\param [in] key: Key that will be used with a hash function to be converted into an index

		\return Index of where key lies
	***********************************************************************************/
	unsigned GetIndex(char const *key) const;

	/*!*********************************************************************************
		\brief Helper function to expand the table if the table exceeds the max load factor
	***********************************************************************************/
	void ExpandTable(void);

	/*!*********************************************************************************
		\brief Helper function to insert an item into the table

		\param [in] key: Key that will be used with a hash function to be converted into an index
		\param [in] data: Data to be inserted
		\param [in] reinserting: This variable will be true when the table expands and stats will not be updated
	***********************************************************************************/
	void InsertItem(char const *key, T const &data, bool reinserting = false) const;

	/*!*********************************************************************************
		\brief Helper function to remove an item from the table

		\param [in] key: Key that will be used with a hash function to be converted into an index
	***********************************************************************************/
	void RemoveItem(char const *key) const;

	/*!*********************************************************************************
		\brief Helper function to find the node based on the key

		\param [in] key: Key that will be used with a hash function to be converted into an index
	***********************************************************************************/
	HashNode Search(char const *key) const;

	HTConfig m_Config{};
	mutable HTStats m_Stats{};
	HashTable m_Head{nullptr};
	bool m_OwnAllocator{false};
};

#include "ChHashTable.cpp"

#endif