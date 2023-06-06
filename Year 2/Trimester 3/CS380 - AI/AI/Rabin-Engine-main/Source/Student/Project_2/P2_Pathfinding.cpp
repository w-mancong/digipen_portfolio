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

GridPos operator+(GridPos const& lhs, GridPos const& rhs)
{
    return lhs += rhs;
}

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
        NewRequest();
        GridPos start = terrain->get_grid_position(request.start);
        goal = terrain->get_grid_position(request.goal);

        Node& startNode = *( map + GetIndex(start) );
        startNode.fx = startNode.gx + GetHx(request, start, goal) * request.settings.weight;
        startNode.info.onList = OPEN_LIST;

        list.Insert(&startNode);
    }

    while (!list.Empty())
    {
        //Node& parentNode = *list.Pop();
        Node& parentNode = *list.Pop();
        GridPos parentPosition = { parentNode.info.row, parentNode.info.col };

        if (IsGoal(parentPosition))
        {
            // TODO: Add nodes into request.path
            request.path.push_back( terrain->get_world_position(parentPosition) );
            Node* pn = parentNode.parent;
            while (pn)
            {
                request.path.push_front( terrain->get_world_position( pn->info.row, pn->info.col ) );
                pn = pn->parent;
            }
            return PathResult::COMPLETE;
        }

        parentNode.info.onList = CLOSE_LIST;
        // Set terrain colour to yellow
        if(request.settings.debugColoring)
            terrain->set_color(parentPosition, Colors::Yellow);

        // Checking with parentNode's neighbours
        ForAllNeighbouringChildNodes(parentNode, request);
        if (request.settings.singleStep)
            return PathResult::PROCESSING;
    }

    return PathResult::IMPOSSIBLE;
}

void AStarPather::NewRequest(void)
{
    memset(map, 0, sizeof(map));
    int w = terrain->get_map_width(), h = terrain->get_map_height();
    int map_size = w * h;
    for (int row{}; row < h; ++row)
    {
        for (int col{}; col < w; ++col)
        {
            size_t index = static_cast<size_t>(row * h + col);
            (map + index)->fx = (map + index)->gx = 0.0f;
            (map + index)->parent   = nullptr;
            (map + index)->info.row = static_cast<short>(row);
            (map + index)->info.col = static_cast<short>(col);
            (map + index)->info.id  = static_cast<short>(index);
            (map + index)->info.onList = NO_LIST;
        }
    }
    list.Clear();
}

void AStarPather::MapChange(void)
{
    NewRequest();

    // TODO: Compute neighbours here
    memset(neighbours, 0, sizeof(neighbours));
    ComputeNeighbours();
}

size_t AStarPather::GetIndex(GridPos pos)
{
    return GetIndex(pos.row, pos.col);
}

size_t AStarPather::GetIndex(int row, int col)
{
    return static_cast<size_t>(row) * terrain->get_map_height() + static_cast<size_t>(col);
}

bool AStarPather::IsGoal(GridPos pos)
{
    return pos.row == goal.row && pos.col == goal.col;
}

bool AStarPather::IsDiagonal(size_t neighbourPosition)
{
    return neighbourPosition == TL || 
           neighbourPosition == TR || 
           neighbourPosition == BL || 
           neighbourPosition == BR;
}

void AStarPather::ComputeNeighbours(void)
{
    for(size_t index{}; index < MAX_SIZE; ++index)
    { 
        GridPos parentPosition = { (map + index)->info.row, (map + index)->info.col };
        size_t neighbourIndex = 0;
        for (size_t i{}; i < MAX_NEIGHBOURS; ++i)
        {
            // Current neighbour's position
            unsigned char neighbour = 0;
            GridPos neighbourPosition = parentPosition + NEIGHBOUR_POSITIONS[i];
            bool isDiagonal = false;

            /********************************************************************************************
            *                                   NEIGHBOUR VALIDATION                                    *
            *********************************************************************************************/
            auto IsNotWall = [](GridPos pos)
            {
                return terrain->is_valid_grid_position(pos) && !terrain->is_wall(pos);
            };

            if (!IsNotWall(neighbourPosition))
                continue;

            if (IsDiagonal(i))
            {
                auto IsValidDiagonalNeighbour = [parentPosition, IsNotWall](size_t index)
                {
                    GridPos pos = parentPosition + NEIGHBOUR_POSITIONS[index];
                    return IsNotWall(pos);
                };

                if (i == TL)
                {
                    if (IsValidDiagonalNeighbour(T) && IsValidDiagonalNeighbour(L))
                        neighbour = TOP_LEFT;
                }
                else if (i == TR)
                {
                    if (IsValidDiagonalNeighbour(T) && IsValidDiagonalNeighbour(R))
                        neighbour = TOP_RIGHT;
                }
                else if (i == BL)
                {
                    if (IsValidDiagonalNeighbour(B) && IsValidDiagonalNeighbour(L))
                        neighbour = BTM_LEFT;
                }
                else if (i == BR)
                {
                    if (IsValidDiagonalNeighbour(B) && IsValidDiagonalNeighbour(R))
                        neighbour = BTM_RIGHT;
                }
                isDiagonal = true;
            }
            else
            {
                if (i == T)
                    neighbour = TOP;
                else if (i == L)
                    neighbour = LEFT;
                else if (i == R)
                    neighbour = RIGHT;
                else if (i == B)
                    neighbour = BTM;
            }

            if (!neighbour) 
                continue;
            neighbours[index] |= neighbour;
        }
    }
}

void AStarPather::InsertNeighbourNode(Node* neighbourNode, Node const& parentNode, PathRequest const& request, size_t diagonalIndex)
{
    GridPos neighbourPosition = { neighbourNode->info.row, neighbourNode->info.col };
    // Compute fx = gx + hx * weight
    float gx = parentNode.gx + (IsDiagonal(diagonalIndex) ? SQRT_2 : 1.0f);
    float fx = gx + GetHx(request, neighbourPosition, goal) * request.settings.weight;

    // If neighbour node is not on list, add it to open list
    if (neighbourNode->info.onList == NO_LIST)
    {
        // set terrain's color to blue
        if (request.settings.debugColoring)
            terrain->set_color(neighbourPosition, Colors::Blue);

        neighbourNode->fx     = fx;
        neighbourNode->gx     = gx;
        neighbourNode->parent = map + parentNode.info.id;
        neighbourNode->info.onList = OPEN_LIST;

        list.Insert(neighbourNode);
    }
    else if (neighbourNode->info.onList != NO_LIST && fx < neighbourNode->fx)
    {
        // QuickArray
        neighbourNode->fx     = fx;
        neighbourNode->gx     = gx;
        neighbourNode->parent = map + parentNode.info.id;
        if (neighbourNode->info.onList == CLOSE_LIST)
            list.Insert(neighbourNode);
        neighbourNode->info.onList = OPEN_LIST;
    }
}

void AStarPather::ForAllNeighbouringChildNodes(Node const& parentNode, PathRequest const& request)
{
    unsigned char n = neighbours[ parentNode.info.id ];
    Node* neighbourNode{ nullptr };
    GridPos neighbourPosition, parentPosition{ parentNode.info.row, parentNode.info.col };

    if (n & TOP_LEFT)
    {
        neighbourPosition = parentPosition + NEIGHBOUR_POSITIONS[TL];
        neighbourNode = map + GetIndex(neighbourPosition);
        InsertNeighbourNode(neighbourNode, parentNode, request, TL);
    }

    if (n & TOP)
    {
        neighbourPosition = parentPosition + NEIGHBOUR_POSITIONS[T];
        neighbourNode = map + GetIndex(neighbourPosition);
        InsertNeighbourNode(neighbourNode, parentNode, request, T);
    }

    if (n & TOP_RIGHT)
    {
        neighbourPosition = parentPosition + NEIGHBOUR_POSITIONS[TR];
        neighbourNode = map + GetIndex(neighbourPosition);
        InsertNeighbourNode(neighbourNode, parentNode, request, TR);
    }

    if (n & LEFT)
    {
        neighbourPosition = parentPosition + NEIGHBOUR_POSITIONS[L];
        neighbourNode = map + GetIndex(neighbourPosition);
        InsertNeighbourNode(neighbourNode, parentNode, request, L);
    }

    if (n & RIGHT)
    {
        neighbourPosition = parentPosition + NEIGHBOUR_POSITIONS[R];
        neighbourNode = map + GetIndex(neighbourPosition);
        InsertNeighbourNode(neighbourNode, parentNode, request, R);
    }

    if (n & BTM_LEFT)
    {
        neighbourPosition = parentPosition + NEIGHBOUR_POSITIONS[BL];
        neighbourNode = map + GetIndex(neighbourPosition);
        InsertNeighbourNode(neighbourNode, parentNode, request, BL);
    }

    if (n & BTM)
    {
        neighbourPosition = parentPosition + NEIGHBOUR_POSITIONS[B];
        neighbourNode = map + GetIndex(neighbourPosition);
        InsertNeighbourNode(neighbourNode, parentNode, request, B);
    }

    if (n & BTM_RIGHT)
    {
        neighbourPosition = parentPosition + NEIGHBOUR_POSITIONS[BR];
        neighbourNode = map + GetIndex(neighbourPosition);
        InsertNeighbourNode(neighbourNode, parentNode, request, BR);
    }
}

float AStarPather::GetHx(PathRequest const& request, GridPos curr, GridPos goal) const
{
    float dx = static_cast<float>(goal.row - curr.row),
          dy = static_cast<float>(goal.col - curr.col);
    float result = 0.0f;

    switch (request.settings.heuristic)
    {
        case Heuristic::OCTILE:
        {
            float min = dx < dy ? dx : dy, max = dx > dy ? dx : dy;
            result = min * SQRT_2 + (max - min); 
            break;
        }

        case Heuristic::CHEBYSHEV:
        {
            result = std::fmax(dx, dy); 
            break;
        }

        case Heuristic::INCONSISTENT:
        {
            if (!((curr.row + curr.col) % 2))
                result = sqrtf(dx * dx + dy * dy);
            else 
                result = 0.0f;
            break;
        }


        case Heuristic::MANHATTAN:
        {
            result = dx + dy; 
            break;
        }

        case Heuristic::EUCLIDEAN:
        {
            result = sqrtf(dx * dx + dy * dy); 
            break;
        }

        default: 
            break;
    }

    return result * (result < 0.0f ? -1.0f : 1.0f);
}