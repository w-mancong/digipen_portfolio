#pragma once
#include "Misc/PathfindingDetails.hpp"

class AStarPather
{
public:
    /* 
        The class should be default constructible, so you may need to define a constructor.
        If needed, you can modify the framework where the class is constructed in the
        initialize functions of ProjectTwo and ProjectThree.
    */
    AStarPather()  = default;
    ~AStarPather() = default;

    /* ************************************************** */
    // DO NOT MODIFY THESE SIGNATURES
    bool initialize();
    void shutdown();
    PathResult compute_path(PathRequest &request);
    /* ************************************************** */

    /*
        You should create whatever functions, variables, or classes you need.
        It doesn't all need to be in this header and cpp, structure it whatever way
        makes sense to you.
    */

private:
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
        SQRT_2 = 1.41421356237f;

    static size_t constexpr MAX_WIDTH{ 40 }, MAX_HEIGHT{ 40 }, MAX_SIZE{ MAX_WIDTH * MAX_HEIGHT }, 
        MAX_NEIGHBOURS{ 8 };

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
    };

    struct Neighbour
    {
        Node const* node;
        unsigned char relativePosition;    // Position of node relative to node
    };

    class OpenList;

    void MapChange(void);
    void ComputeNeighbours(void);
    void ResetMap(void);
    float GetHx(PathRequest const& request, GridPos curr, GridPos goal) const;
    bool Check(Node const& node, unsigned char list) const;
    void SetListStatus(Node& node, unsigned char list);
    GridPos MakeGrid(Node const& node) const;
    Node& GetNode(GridPos pos);
    Node const& GetNode(GridPos pos) const;
    Node& GetNode(int row, int col);
    Node const& GetNode(int row, int col) const;
    Node& GetNode(size_t id);
    Node const& GetNode(size_t id) const;
    size_t GetArrayPosition(GridPos pos) const;
    size_t GetArrayPosition(int row, int col) const;
    bool IsGoal(Node const& node);
    bool IsDiagonal(unsigned char relativePosition);

    Node map[MAX_SIZE];
    Neighbour neighbours[MAX_SIZE][MAX_NEIGHBOURS];
    OpenList* list;
    GridPos goal;
};