#pragma once

static unsigned char constexpr const
NO_LIST     = 0b00,
OPEN_LIST   = 0b01,
CLOSE_LIST  = 0b10;     // variables used to check if node is on the list
static unsigned char constexpr const
TOP_LEFT    = 0b00000001,
TOP         = 0b00000010,
TOP_RIGHT   = 0b00000100,
LEFT        = 0b00001000,
RIGHT       = 0b00010000,
BTM_LEFT    = 0b00100000,
BTM         = 0b01000000,
BTM_RIGHT   = 0b10000000;    // variables used to check neighbour's position relative to current cell
static float constexpr const
SQRT_2      = 1.41421356237f;

static size_t constexpr MAX_WIDTH{ 40 }, MAX_HEIGHT{ 40 }, MAX_SIZE{ MAX_WIDTH * MAX_HEIGHT }, MAX_NEIGHBOURS{ 8 };

struct NodeInfo
{
    unsigned short row : 6;     // Node's x position
    unsigned short col : 6;     // Node's y position
    unsigned short onList : 2;  // Check to see if node is on list
    unsigned short id : 12;     // ID of Node (to index it by finding it through the array)

    NodeInfo() : row{ 0 }, col{ 0 }, onList{ NO_LIST }, id{ 0 } {}
};

struct Node
{
    Node* parent{};               // Node's parent
    float finalCost{};            // Node's final cost
    float givenCost{};            // Node's given cost
    NodeInfo info{};              // Contains grid position and a value to check if node is on list

    bool operator<(Node const& rhs)
    {
        return finalCost < rhs.finalCost;
    }
};

class MinHeap
{
public:
	MinHeap(void);
    ~MinHeap(void);

    void Insert(Node* node);
    void Heapify(void);
    Node* Pop(void);
    void Clear(void);
    bool Empty(void) const;
    size_t size() const;

private:
    size_t parent(size_t i) const;
    size_t left(size_t i) const;
    size_t right(size_t i) const;
    void Heapify(size_t i);

    Node** arr{ nullptr };
    size_t heapSize{};

    friend std::ostream& operator<<(std::ostream& os, MinHeap const& p);
};

std::ostream& operator<<(std::ostream& os, MinHeap const& p);