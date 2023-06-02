#include <pch.h>
#include "Projects/ProjectTwo.h"
#include "P2_Pathfinding.h"

class AStarPather::OpenList
{
    OpenList();
    ~OpenList();

    struct Bucket
    {
        Node** node{ nullptr };  // List of pointers to nodes
        size_t size{ 0 };        // store the total number of items in this bucket
    };

    Bucket* list;
};

#pragma region Extra Credit
bool ProjectTwo::implemented_floyd_warshall()
{
    return false;
}

bool ProjectTwo::implemented_goal_bounding()
{
    return false;
}

bool ProjectTwo::implemented_jps_plus()
{
    return false;
}
#pragma endregion

bool AStarPather::initialize()
{
    // handle any one-time setup requirements you have

    /*
        If you want to do any map-preprocessing, you'll need to listen
        for the map change message.  It'll look something like this:

        Callback cb = std::bind(&AStarPather::your_function_name, this);
        Messenger::listen_for_message(Messages::MAP_CHANGE, cb);

        There are other alternatives to using std::bind, so feel free to mix it up.
        Callback is just a typedef for std::function<void(void)>, so any std::invoke'able
        object that std::function can wrap will suffice.
    */

    Callback cb = std::bind(&AStarPather::MapChange, this);
    Messenger::listen_for_message(Messages::MAP_CHANGE, cb);

    return true; // return false if any errors actually occur, to stop engine initialization
}

void AStarPather::shutdown()
{
    /*
        Free any dynamically allocated memory or any other general house-
        keeping you need to do during shutdown.
    */
}

PathResult AStarPather::compute_path(PathRequest &request)
{
    /*
        This is where you handle pathing requests, each request has several fields:

        start/goal - start and goal world positions
        path - where you will build the path upon completion, path should be
            start to goal, not goal to start
        heuristic - which heuristic calculation to use
        weight - the heuristic weight to be applied
        newRequest - whether this is the first request for this path, should generally
            be true, unless single step is on

        smoothing - whether to apply smoothing to the path
        rubberBanding - whether to apply rubber banding
        singleStep - whether to perform only a single A* step
        debugColoring - whether to color the grid based on the A* state:
            closed list nodes - yellow
            open list nodes - blue

            use terrain->set_color(row, col, Colors::YourColor);
            also it can be helpful to temporarily use other colors for specific states
            when you are testing your algorithms

        method - which algorithm to use: A*, Floyd-Warshall, JPS+, or goal bounding,
            will be A* generally, unless you implement extra credit features

        The return values are:
            PROCESSING - a path hasn't been found yet, should only be returned in
                single step mode until a path is found
            COMPLETE - a path to the goal was found and has been built in request.path
            IMPOSSIBLE - a path from start to goal does not exist, do not add start position to path

        // Just sample code, safe to delete
        //GridPos start = terrain->get_grid_position(request.start);
        //GridPos goal = terrain->get_grid_position(request.goal);
        //terrain->set_color(start, Colors::Orange);
        //terrain->set_color(goal, Colors::Orange);
        //request.path.push_back(request.start);
        //request.path.push_back(request.goal);
    */

    // WRITE YOUR CODE HERE
    if (request.newRequest)
    {
        ResetMap();
        // clear open list
        // push start node into the open list with cost f(x) = g(x) + h(x) * weight
    }
    
    return PathResult::COMPLETE;
}

void AStarPather::MapChange(void)
{
    // get each node's neighbour
    memset(neighbours, 0, sizeof(neighbours));
    ComputeNeighbours();
}

void AStarPather::ComputeNeighbours(void)
{
    size_t const WIDTH  = static_cast<size_t>( terrain->get_map_width() ),
                 HEIGHT = static_cast<size_t>( terrain->get_map_height() );

    auto IsDiagonal = [](unsigned char relativePosition)
    {
        return relativePosition & TOP_LEFT  ||
               relativePosition & TOP_RIGHT ||
               relativePosition & BTM_LEFT  ||
               relativePosition & BTM_RIGHT;
    };

    auto ShouldBeNeighbour = [](unsigned char relativePosition, GridPos pos)
    {
        static size_t constexpr const TOP{ 0 }, LEFT{ 1 }, RIGHT{ 2 }, BTM{ 3 };
        GridPos wall[4]{ { pos.row + 1, pos.col },      // Top
                         { pos.row, pos.col - 1 },      // Left
                         { pos.row, pos.col + 1 },      // Right
                         { pos.row - 1, pos.col } };    // Btm

        auto IsWall = [](GridPos pos)
        {
            return terrain->is_valid_grid_position(pos) && terrain->is_wall(pos);
        };

        if (relativePosition & TOP_LEFT)
        {
            if ( IsWall(wall[TOP]) || IsWall(wall[LEFT]) )
                return false;
        }

        if (relativePosition & TOP_RIGHT)
        {
            if (IsWall(wall[TOP]) || IsWall(wall[RIGHT]))
                return false;
        }

        if (relativePosition & BTM_LEFT)
        {
            if (IsWall(wall[BTM]) || IsWall(wall[LEFT]))
                return false;
        }

        if (relativePosition & BTM_RIGHT)
        {
            if (IsWall(wall[BTM]) || IsWall(wall[RIGHT]))
                return false;
        }

        return true;
    };

    auto GetNeighbours = [this, IsDiagonal, ShouldBeNeighbour](int i, int j)
    {
        size_t index = 0; unsigned char relativePosition = 0;
        int const nodeIndex = static_cast<int>( GetArrayPosition( i, j ) );

        for (int row{ 1 }; row >= -1; --row)
        {
            for (int col{ -1 }; col <= 1; ++col)
            {
                if (row == 0 && col == 0)
                    continue;

                ++relativePosition;
                GridPos pos{ i + row, j + col };
                if ( !terrain->is_valid_grid_position( pos ) || terrain->is_wall(pos) )
                    continue;

                if ( IsDiagonal( 0b1 << (relativePosition - 1) ) && !ShouldBeNeighbour( 0b1 << (relativePosition - 1), { i , j } ) )
                        continue;

                neighbours[nodeIndex][index].neighbour = 0b1 << (relativePosition - 1);
                neighbours[nodeIndex][index++].node = &map[ GetArrayPosition(i + row, j + col) ];
            }
        }
    };

    for (int i{}; i < HEIGHT; ++i)
    {
        for (int j{}; j < WIDTH; ++j)
            GetNeighbours(i, j);
    }
}

void AStarPather::ResetMap(void)
{
    memset(map, 0, sizeof(map));
    for (size_t i{}; i < MAX_SIZE; ++i)
        (map + i)->info.id = i;

}

float AStarPather::GetHx(Heuristic heu, GridPos curr, GridPos goal) const
{
    float dx = static_cast<float>(curr.row - goal.row),
          dy = static_cast<float>(curr.col - goal.col);

    switch (heu)
    {
        case Heuristic::OCTILE:
        {
            float min = std::fmin(dx, dy), max = std::fmax(dx, dy);
            return min * SQRT_2 + (max - min);
        }

        case Heuristic::CHEBYSHEV:
            return std::fmax(dx, dy);

        case Heuristic::INCONSISTENT:
        {
            if ( ! ( (curr.row + curr.col) % 2 ) )
                return sqrtf(dx * dx + dy * dy);
            return 0.0f;
        }

        case Heuristic::MANHATTAN:
            return dx + dy;

        case Heuristic::EUCLIDEAN:
            return sqrtf( dx * dx + dy * dy );

        default:
            break;
    }

    return 0.0f;
}

bool AStarPather::Check(Node const& node, unsigned char list) const
{
    return node.info.onList & list;
}

void AStarPather::SetListStatus(Node& node, unsigned char list)
{
    node.info.onList = list;
}

GridPos AStarPather::MakeGrid(Node const& node) const
{
    return { static_cast<int>(node.info.row), static_cast<int>(node.info.col) };
}

AStarPather::Node& AStarPather::GetNode(int row, int col)
{
    return const_cast<Node&>( const_cast<AStarPather const&>(*this).GetNode(row, col) );
}

AStarPather::Node const& AStarPather::GetNode(int row, int col) const
{
    return *( map + GetArrayPosition(row, col) );
}

AStarPather::Node& AStarPather::GetNode(size_t id)
{
    return const_cast<Node&>( const_cast<AStarPather const&>(*this).GetNode(id) );
}

AStarPather::Node const& AStarPather::GetNode(size_t id) const
{
    return *(map + id);
}

size_t AStarPather::GetArrayPosition(int row, int col) const
{
    return static_cast<size_t>(row) * terrain->get_map_width() + static_cast<size_t>(col);
}

AStarPather::OpenList::OpenList()
{
    list = new Bucket[ (MAX_SIZE << 1) ];

}

AStarPather::OpenList::~OpenList()
{
    delete[] list;
}
