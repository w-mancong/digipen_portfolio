#include <pch.h>
#include "Projects/ProjectTwo.h"
#include "P2_Pathfinding.h"

class AStarPather::OpenList
{
public:
    OpenList(size_t subDivisions = 5);
    ~OpenList();

    void Insert(Node* node);
    void Reinsert(Node* node, float oldCost);
    Node& Pop();
    void Clear();
    bool Empty() const;

private:
    static size_t constexpr const DEFAULT_SIZE{ 128 };

    struct SubBucket
    {
        SubBucket() : capacity{ DEFAULT_SIZE } {};
        ~SubBucket() = default;
         
        size_t capacity{ 0 };     // store the capacity for this SubBucket

        void Init(Node const** node);
        void Insert(Node const* node);
        Node& Pop();
        bool Remove(Node const& node);
        void Clear();
        //void Sort();

        Node const ** nodes{ nullptr };  // List of pointers to nodes
        size_t size{ 0 };                // store the total number of items in this bucket
    };

    struct Bucket
    {
        void Init(SubBucket* bucket);
        void Insert(Node const* node, size_t index);
        Node& Pop(bool& isEmpty, size_t subDivisions);
        void Remove(Node const& node, size_t index);
        void Clear(size_t subDivisions);

        size_t size{};
        SubBucket* buckets{ nullptr };
    };

    Bucket* buckets{};
    SubBucket* subBuckets{};
    Node const** subBucketNodes{};
    std::set<size_t> indexSet;
    size_t smallestIndex{ std::numeric_limits<size_t>::max() }; // Used to store the index to the smallest f(x). 
    size_t totalNodes{};
    size_t const subDivisions;
    float const divisions;
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

    list = new OpenList{};
    memset(map, 0, sizeof(map));

    return true; // return false if any errors actually occur, to stop engine initialization
}

void AStarPather::shutdown()
{
    /*
        Free any dynamically allocated memory or any other general house-
        keeping you need to do during shutdown.
    */
    delete list;
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
        // clear open list
        // push start node into the open list with cost f(x) = g(x) + h(x) * weight
        ResetMap();
        goal          = terrain->get_grid_position(request.goal);
        GridPos start = terrain->get_grid_position(request.start);
        Node& node = GetNode( start );
        node.finalCost = node.givenCost + GetHx(request, start, goal) * request.settings.weight;
        list->Insert(&node);
    }
    
    while (!list->Empty())
    {
        // Pop cheapest node off the list
        Node& parentNode = list->Pop();

        // if parent node is goal, path found
        if (IsGoal(parentNode))
        {
            // assign the path to request.path
            return PathResult::COMPLETE;
        }

        // Place parent node on close list
        parentNode.info.onList = CLOSE_LIST;
        terrain->set_color(parentNode.info.row, parentNode.info.col, Colors::Yellow);
         // Retrieve all valid neighbours from parentNode
        GridPos pos = MakeGrid(parentNode);
        size_t parentIndex = GetArrayPosition(pos);

        for (size_t i{}; i < MAX_NEIGHBOURS; ++i)
        {
            Neighbour const& child = *(*(neighbours + parentIndex) + i);
            if (!child.relativePosition)
                break;

            // new gx cost for current node
            float gx = parentNode.givenCost + (IsDiagonal(child.relativePosition) ? SQRT_2 : 1.0f);
            float fx = gx + GetHx(request, pos, goal) * request.settings.weight;

            Node childNode = map[child.node->info.id];

            // if child node not on list, add to open list
            if (childNode.info.onList == NO_LIST)
            {
                childNode.finalCost = fx;
                childNode.givenCost = gx;
                list->Insert(&childNode);
            }
            // if child node is either on open/close list, reinsert the node into the correct bucket
            else if (child.node->info.onList != NO_LIST && fx < child.node->finalCost)
            {
                float oldCost = child.node->finalCost;
                childNode.finalCost = fx;
                childNode.givenCost = gx;
                list->Reinsert(&childNode, oldCost);
            }
        }
        //if (request.settings.singleStep)
        //    return PathResult::PROCESSING;
    }
    return PathResult::IMPOSSIBLE;
}

void AStarPather::MapChange(void)
{
    ResetMap();
    // get each node's neighbour
    memset(neighbours, 0, sizeof(neighbours));
    ComputeNeighbours();
}

void AStarPather::ComputeNeighbours(void)
{
    size_t const WIDTH  = static_cast<size_t>( terrain->get_map_width() ),
                 HEIGHT = static_cast<size_t>( terrain->get_map_height() );

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

    auto GetNeighbours = [this, ShouldBeNeighbour](int i, int j)
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

                neighbours[nodeIndex][index].relativePosition = 0b1 << (relativePosition - 1);
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
    for (int i{}; i < terrain->get_map_height(); ++i)
    {
        for (int j{}; j < terrain->get_map_width(); ++j)
        {
            size_t index = GetArrayPosition(i, j);
            (map + index)->finalCost   = (map + index)->givenCost = 0.0f;
            (map + index)->parent      = nullptr;
            (map + index)->info.row    = static_cast<unsigned short>(i);
            (map + index)->info.col    = static_cast<unsigned short>(j);
            (map + index)->info.id     = static_cast<unsigned short>(index);
            (map + index)->info.onList = NO_LIST;
        }
    }
    list->Clear();
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
            float min = std::fmin(dx, dy), max = std::fmax(dx, dy);
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

    return std::abs(result);
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

AStarPather::Node& AStarPather::GetNode(GridPos pos)
{
    return GetNode(pos.row, pos.col);
}

AStarPather::Node const& AStarPather::GetNode(GridPos pos) const
{
    return GetNode(pos.row, pos.col);
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

size_t AStarPather::GetArrayPosition(GridPos pos) const
{
    return GetArrayPosition(pos.row, pos.col);
}

size_t AStarPather::GetArrayPosition(int row, int col) const
{
    return static_cast<size_t>(row) * terrain->get_map_width() + static_cast<size_t>(col);
}

bool AStarPather::IsGoal(Node const& node)
{
    return node.info.row == goal.row && node.info.col == goal.col;
}

bool AStarPather::IsDiagonal(unsigned char relativePosition)
{
    return relativePosition & TOP_LEFT  ||
           relativePosition & TOP_RIGHT ||
           relativePosition & BTM_LEFT  ||
           relativePosition & BTM_RIGHT;
}

/******************************************************************************************
                                   OPEN_LIST
******************************************************************************************/
AStarPather::OpenList::OpenList(size_t subDivisions) : subDivisions{ subDivisions }, divisions { 1.0f / static_cast<float>(subDivisions) }
{
    size_t const LIST_SIZE = MAX_SIZE << 1,
                 SUB_SIZE  = subDivisions;

    buckets         = new Bucket[LIST_SIZE]{};
    subBuckets      = new SubBucket[LIST_SIZE * SUB_SIZE] {};
    subBucketNodes  = const_cast<Node const**>( new Node * [LIST_SIZE * SUB_SIZE * DEFAULT_SIZE]{} );

    for (size_t i{}; i < LIST_SIZE; ++i)
        (buckets + i)->Init( subBuckets + i * SUB_SIZE );

    for (size_t i{}; i < LIST_SIZE * SUB_SIZE; ++i)
        (subBuckets + i)->Init( subBucketNodes + i * DEFAULT_SIZE );
}

AStarPather::OpenList::~OpenList()
{
    delete[] buckets;
    delete[] subBuckets;
    delete[] subBucketNodes;
}

void AStarPather::OpenList::Insert(Node* node)
{                                              // x.abcd -> 2.7643 
    float whole = std::floor(node->finalCost), // Stores 2.0
          decimals = node->finalCost - whole;  // Stores 0.7643

    size_t const bucketIndex    = static_cast<size_t>(whole), 
                 subBucketIndex = static_cast<size_t>(decimals / divisions);

    node->info.onList = OPEN_LIST;
    terrain->set_color(node->info.row, node->info.col, Colors::Blue);

    (buckets + bucketIndex)->Insert(node, subBucketIndex);
    if (bucketIndex < smallestIndex)
        smallestIndex = bucketIndex;

    //indexSet.insert(bucketIndex);

    ++totalNodes;
}

void AStarPather::OpenList::Reinsert(Node* node, float oldCost)
{
    float oldWhole    = std::floor(oldCost),
          oldDecimals = oldCost - oldWhole;

    size_t const oldBucketIndex     = static_cast<size_t>(oldWhole),
                 oldSubBucketIndex  = static_cast<size_t>(oldDecimals / divisions);

                                                      // x.abcd -> 2.7643 
    float whole       = std::floor(node->finalCost),  // Stores 2.0
          decimals    = node->finalCost - whole;      // Stores 0.7643

    size_t const bucketIndex    = static_cast<size_t>(whole),
                 subBucketIndex = static_cast<size_t>(decimals / divisions);

    if (oldSubBucketIndex == subBucketIndex)
        return;
    node->info.onList = OPEN_LIST;
    terrain->set_color(node->info.row, node->info.col, Colors::Blue);

    (buckets + oldBucketIndex)->Remove(*node, oldSubBucketIndex);
    (buckets + bucketIndex)->Insert(node, subBucketIndex);
    if (bucketIndex < smallestIndex)
        smallestIndex = bucketIndex;
}

AStarPather::Node& AStarPather::OpenList::Pop()
{
    bool isEmpty{ false };
    std::cout << "smallestIndex: " << smallestIndex;
    Node& node = (buckets + smallestIndex)->Pop(isEmpty, subDivisions);
    if (isEmpty)
    {
        size_t i = smallestIndex;
        smallestIndex = std::numeric_limits<size_t>::max();
        while (i < (MAX_SIZE << 1))
        {
            if (!(buckets + ++i)->size)
                continue;

            smallestIndex = i;
            break;
        }
    }
    --totalNodes;
    return node;
}

void AStarPather::OpenList::Clear()
{
    smallestIndex = std::numeric_limits<size_t>::max();

    for (size_t i{}; i < (MAX_SIZE << 1); ++i)
        (buckets + i)->Clear(subDivisions);
    totalNodes = 0;
}

bool AStarPather::OpenList::Empty() const
{
    return totalNodes == 0;
}

/******************************************************************************************
                                       BUCKET
******************************************************************************************/
void AStarPather::OpenList::Bucket::Init(SubBucket* bucket)
{
    buckets = bucket;
}

void AStarPather::OpenList::Bucket::Insert(Node const* node, size_t index)
{
    ++size;
    (buckets + index)->Insert(node);
}

AStarPather::Node& AStarPather::OpenList::Bucket::Pop(bool& isEmpty, size_t subDivisions)
{
    size_t index{};
    while ( index < subDivisions && !(buckets + index)->size )
        ++index;
    std::cout << " sub_div index: " << index << std::endl;
    isEmpty = !(--size);
    return (buckets + index)->Pop();
}

void AStarPather::OpenList::Bucket::Remove(Node const& node, size_t index)
{
    if ((buckets + index)->Remove(node)) --size;
}

void AStarPather::OpenList::Bucket::Clear(size_t subDivisions)
{
    for (size_t i{}; i < subDivisions; ++i)
        (buckets + i)->Clear();
    size = 0;
}

/******************************************************************************************
                                   SUB-BUCKET
******************************************************************************************/
void AStarPather::OpenList::SubBucket::Init(Node const** node)
{
    nodes = node;
}

void AStarPather::OpenList::SubBucket::Insert(Node const* node)
{
#ifdef _DEBUG
    assert(size < capacity && "Increase DEFAULT_SIZE or make more sub divisions");
#endif
    *(nodes + size++) = node;
    Node* nonConst = const_cast<Node*>(*nodes);
    // Arranging it from biggest to smallest
    // Can just return the last element in the array as it is for sure the smallest
    std::sort(nonConst, (nonConst + size), [](Node const& lhs, Node const& rhs)
        {
            return lhs.finalCost > rhs.finalCost;
        });
}

AStarPather::Node& AStarPather::OpenList::SubBucket::Pop()
{
    return const_cast<Node&>( *( *(nodes + --size) ) );
}

bool AStarPather::OpenList::SubBucket::Remove(Node const& node)
{
    bool removed = false;
    for (size_t i{}; i < size; ++i)
    {
        if ((*(nodes + i))->info.id != node.info.id) 
            continue;

        for (size_t j{ i }; j < size; ++j)
        {
            if (!*(nodes + j))
                break;
            *(nodes + j) = *(nodes + j + 1);
        }
        removed = true;
        break;
    }
    if(removed)
        --size;
    return removed;
}

void AStarPather::OpenList::SubBucket::Clear()
{
    for (size_t i{}; i < DEFAULT_SIZE; ++i)
        *(nodes + i) = nullptr;
    size = 0;
}

//void AStarPather::OpenList::SubBucket::Sort()
//{
//    //Node* nonConst = const_cast<Node*>(*nodes);
//    //// Arranging it from biggest to smallest
//    //// Can just return the last element in the array as it is for sure the smallest
//    //std::sort(nonConst, (nonConst + size), [](Node const& lhs, Node const& rhs)
//    //    {
//    //        return lhs.finalCost > rhs.finalCost;
//    //    });
//}