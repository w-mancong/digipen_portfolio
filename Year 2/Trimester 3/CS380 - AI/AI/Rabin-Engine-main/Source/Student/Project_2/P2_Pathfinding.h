#pragma once
#include "Misc/PathfindingDetails.hpp"
#include "MinHeap.h"

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
    struct Neighbour
    {
        Node const* node;
        unsigned char relativePosition;    // Position of node relative to node
    };

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
    Neighbour** neighbours;
    MinHeap list;
    GridPos goal;
};