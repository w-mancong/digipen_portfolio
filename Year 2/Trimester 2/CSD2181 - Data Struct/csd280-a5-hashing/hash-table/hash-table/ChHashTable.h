//---------------------------------------------------------------------------
#ifndef CHHASHTABLEH
#define CHHASHTABLEH
//---------------------------------------------------------------------------

#include <string>
#include "ObjectAllocator.h"
#include "support.h"

// client-provided hash function: takes a key and table size,
// returns an index in the table.
using HASHFUNC = unsigned (*)(const char*, unsigned);

// Max length of our "string" keys
const unsigned MAX_KEYLEN = 10;

class HashTableException
{
private:
	int error_code_;
	std::string message_;

public:
	HashTableException(int ErrCode, const std::string& Message) :
		error_code_(ErrCode), message_(Message) {};

	virtual ~HashTableException() {
	}

	virtual int code() const {
		return error_code_;
	}

	virtual const char* what() const {
		return message_.c_str();
	}
	enum HASHTABLE_EXCEPTION { E_ITEM_NOT_FOUND, E_DUPLICATE, E_NO_MEMORY };
};


// HashTable statistical info
struct HTStats
{
	HTStats(void) : Count_(0), TableSize_(0), Probes_(0), Expansions_(0),
		HashFunc_(nullptr), Allocator_(nullptr) {};
	unsigned Count_;      // Number of elements in the table
	unsigned TableSize_;  // Size of the table (total slots)
	unsigned Probes_;     // Number of probes performed
	unsigned Expansions_; // Number of times the table grew
	HASHFUNC HashFunc_;   // Pointer to primary hash function
	ObjectAllocator* Allocator_; // The allocator in use (may be 0)
};

template <typename T>
class ChHashTable
{
public:
	using FREEPROC = void (*)(T);

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
			FreeProc_(FreeProc) {}

		unsigned InitialTableSize_{};
		HASHFUNC HashFunc_{};
		double MaxLoadFactor_{};
		double GrowthFactor_{};
		FREEPROC FreeProc_{};
	};

	// Nodes that will hold the key/data pairs
	struct ChHTNode
	{
		char Key[MAX_KEYLEN]; // Key is a string
		T Data{};               // Client data
		ChHTNode* Next{ nullptr };
		ChHTNode(const T& data) : Data(data) {}; // constructor
	};

	// Each list has a special head pointer
	struct ChHTHeadNode
	{
		ChHTNode* Nodes;
		ChHTHeadNode() : Nodes(0), Count(0) {};
		int Count; // For testing
	};

	using HashNode = ChHTHeadNode*;

	// ObjectAllocator: the usual.
	// Config: the configuration for the hash table.
	ChHashTable(HTConfig const& Config, ObjectAllocator* allocator = nullptr);
	~ChHashTable();

	// Insert a key/data pair into table. Throws an exception if the
	// insertion is unsuccessful.(E_DUPLICATE, E_NO_MEMORY)
	void insert(char const* Key, T const& Data);

	// Delete an item by key. Throws an exception if the key doesn't exist.
	// (E_ITEM_NOT_FOUND)
	void remove(char const* Key);

	// Find and return data by key. throws exception if key doesn't exist.
	// (E_ITEM_NOT_FOUND)
	const T& find(char const* Key) const;

	// Removes all items from the table (Doesn't deallocate table)
	void clear();

	// Allow the client to peer into the data. Returns a struct that contains 
	// information on the status of the table for debugging and testing. 
	// The struct is defined in the header file.
	HTStats GetStats() const;
	ChHTHeadNode const* GetTable() const;

private:
	// Private fields and methods...

	HTConfig m_Config{};
	HTStats m_Stats{};
	HashNode m_Head{ nullptr };
	bool m_OwnAllocator{ false };
};

//#include "ChHashTable.cpp"
template <typename T>
ChHashTable<T>::ChHashTable(HTConfig const& Config, ObjectAllocator* allocator) : m_Config{ Config }
{
	m_Stats.Allocator_ = allocator;
	if (!allocator)
	{
		m_Stats.Allocator_ = new ObjectAllocator(sizeof(T), OAConfig());
		m_OwnAllocator = true;
	}
}

template <typename T>
ChHashTable<T>::~ChHashTable(void)
{
	if (m_OwnAllocator)
	{
		delete m_Stats.Allocator_;
		m_Stats.Allocator_ = nullptr;
	}
}

template <typename T>
void ChHashTable<T>::insert(char const* Key, T const& Data)
{

}

template <typename T>
void ChHashTable<T>::remove(char const* Key)
{

}

template <typename T>
const T& ChHashTable<T>::find(char const* Key) const
{
	return {};
}

template <typename T>
void ChHashTable<T>::clear()
{

}

template <typename T>
HTStats ChHashTable<T>::GetStats() const
{
	return m_Stats;
}

template <typename T>
typename ChHashTable<T>::ChHTHeadNode const* ChHashTable<T>::GetTable() const
{
	return m_Head;
}

#endif
