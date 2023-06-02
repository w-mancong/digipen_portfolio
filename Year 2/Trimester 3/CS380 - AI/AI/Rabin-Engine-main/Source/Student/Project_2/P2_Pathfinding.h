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

    struct NodeInfo
    {
        unsigned short x : 6;       // Node's x position
        unsigned short y : 6;       // Node's y position
        unsigned short onList : 2;  // Check to see if node is on list
    };

    struct Node
    {
        Node* parent;           // Node's parent
        float finalCost;        // Node's final cost
        float givenCost;        // Node's given cost
        NodeInfo info;          // Contains grid position and a value to check if node is on list
    };

    float GetHx(Heuristic heu, GridPos curr, GridPos goal);
    bool Check(Node const& node, unsigned char list);
    void SetListStatus(Node& node, unsigned char list);
    GridPos MakeGrid(Node const& node);
};