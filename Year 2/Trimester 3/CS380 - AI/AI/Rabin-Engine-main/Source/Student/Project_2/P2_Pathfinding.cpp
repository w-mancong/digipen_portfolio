#include <pch.h>
#include "Projects/ProjectTwo.h"
#include "P2_Pathfinding.h"

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
    */

    // WRITE YOUR CODE HERE

    
    // Just sample code, safe to delete
    //GridPos start = terrain->get_grid_position(request.start);
    //GridPos goal = terrain->get_grid_position(request.goal);
    //terrain->set_color(start, Colors::Orange);
    //terrain->set_color(goal, Colors::Orange);
    //request.path.push_back(request.start);
    //request.path.push_back(request.goal);
    return PathResult::COMPLETE;
}

float AStarPather::GetHx(Heuristic heu, GridPos curr, GridPos goal)
{
    float dx = static_cast<float>(curr.row - goal.row),
          dy = static_cast<float>(curr.col - goal.col);

    switch (heu)
    {
        case Heuristic::OCTILE:
        {
            float constexpr const SQRT2 = 1.41421356237f;
            float min = std::fmin(dx, dy), max = std::fmax(dx, dy);
            return min * SQRT2 + (max - min);
        }

        case Heuristic::CHEBYSHEV:
            return std::fmax(dx, dy);

        case Heuristic::INCONSISTENT:
        {
            if ( !(curr.row + curr.col) % 2 )
                return sqrtf(dx * dx + dy * dy);
            return 0.0f;
        }

        case Heuristic::MANHATTAN:
            return dx + dy; break;

        case Heuristic::EUCLIDEAN:
            return sqrtf( dx * dx + dy * dy );

        default:
            break;
    }

    return 0.0f;
}

bool AStarPather::Check(Node const& node, unsigned char list)
{
    return node.info.onList & list;
}

void AStarPather::SetListStatus(Node& node, unsigned char list)
{
    node.info.onList = list;
}

GridPos AStarPather::MakeGrid(Node const& node)
{
    return { static_cast<int>(node.info.x), static_cast<int>(node.info.y) };
}
