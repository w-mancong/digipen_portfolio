//---------------------------------------------------------------------------
#ifndef ALGRAPH_H
#define ALGRAPH_H
//---------------------------------------------------------------------------
#include <vector>
#include <queue>
#include <algorithm>

struct DijkstraInfo
{
	unsigned cost;
	std::vector<unsigned> path;
};

struct AdjacencyInfo
{
	unsigned id;
	unsigned weight;
};

using ALIST = std::vector< std::vector<AdjacencyInfo> >;

class ALGraph
{
public:
	ALGraph(unsigned size);
	~ALGraph(void);
	void AddDEdge(unsigned source, unsigned destination, unsigned weight);
	void AddUEdge(unsigned node1, unsigned node2, unsigned weight);

	std::vector<DijkstraInfo> Dijkstra(unsigned start_node) const;
	ALIST GetAList(void) const;

private:
	static constexpr const unsigned INF = std::numeric_limits<unsigned>::max();
	struct AdjInfo
	{
		bool operator()(AdjInfo const* lhs, AdjInfo const* rhs) const;

		unsigned id{ 0 };
		unsigned cost{ 0 };
		unsigned prev{ 0 };
	};

	using MinHeap = std::priority_queue < AdjInfo*, std::vector<AdjInfo*>, AdjInfo >;

	//using MinHeap = std::vector<AdjInfo*>;

	// Other private fields and methods
	ALIST list{};
};

#endif
