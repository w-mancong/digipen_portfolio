#pragma once
#include "Misc/PathfindingDetails.hpp"
#include "Node.h"
#include "MinHeap.h"
#include "OpenList.h"

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
        unsigned short id{ std::numeric_limits<unsigned short>::max() };
        bool isDiagonal{};
        bool isNeighbour{};
    };

    void NewRequest(void);
    void MapChange(void);
    size_t GetIndex(GridPos pos);
    size_t GetIndex(int row, int col);

    bool IsGoal(GridPos pos);
    //GridPos MakeGrid(Node const& node);

    bool IsDiagonal(size_t neighbourPosition);
    void ComputeNeighbours(void);

    float GetHx(PathRequest const& request, GridPos curr, GridPos goal) const;

    Node map[MAX_SIZE]{};
    Neighbour neighbours[MAX_SIZE][8]{};
    //MinHeap list;
    GridPos goal;

    OpenList a;
};