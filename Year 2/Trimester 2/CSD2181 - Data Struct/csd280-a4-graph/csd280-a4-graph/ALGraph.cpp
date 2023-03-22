#include "ALGraph.h"

ALGraph::ALGraph(unsigned size)
{
	list.resize(static_cast<size_t>(size));
}

ALGraph::~ALGraph(void)
{

}

void ALGraph::AddDEdge(unsigned source, unsigned destination, unsigned weight)
{
	size_t const index = static_cast<size_t>(source) - 1;
	list[index].emplace_back( AdjacencyInfo{ destination, weight } );
	std::sort(list[index].begin(), list[index].end(), [](AdjacencyInfo const& lhs, AdjacencyInfo const& rhs)
		{
			if (lhs.weight == rhs.weight)
				return lhs.id < rhs.id;
			return lhs.weight < rhs.weight;
		});
}

void ALGraph::AddUEdge(unsigned node1, unsigned node2, unsigned weight)
{
	AddDEdge(node1, node2, weight);
	AddDEdge(node2, node1, weight);
}

std::vector<DijkstraInfo> ALGraph::Dijkstra(unsigned start_node) const
{
	std::vector<AdjInfo> nodes( list.size() );
	MinHeap Q;
	for (size_t i{}; i < nodes.size(); ++i)
	{
		unsigned id = static_cast<unsigned>(i) + 1;
		nodes[i] = { id, INF, 0 };
	}

	nodes[start_node - 1].cost = 0;
	for(size_t i{}; i < nodes.size(); ++i)
		Q.emplace(&nodes[i]);

	while (!Q.empty())
	{
		AdjInfo* u = Q.top();
		for (AdjacencyInfo const& v : list[u->id - 1])
		{
			if (nodes[v.id - 1].cost > u->cost + v.weight)
			{
				nodes[v.id - 1].cost = u->cost + v.weight;
				nodes[v.id - 1].prev = u->id;
			}
		}
		Q.pop();
	}

	std::vector<DijkstraInfo> res( list.size() );
	for (size_t i{}; i < res.size(); ++i)
	{
		res[i].cost = nodes[i].cost;
		unsigned id = nodes[i].id;
		while (id != start_node)
		{
			res[i].path.emplace_back( nodes[id - 1].id );
			id = nodes[id - 1].prev;
		}
		res[i].path.emplace_back( start_node );
		std::reverse( res[i].path.begin(), res[i].path.end() );
	}

	return res;
}

ALIST ALGraph::GetAList(void) const
{
	return list;
}

bool ALGraph::AdjInfo::operator()(AdjInfo const* lhs, AdjInfo const* rhs) const
{
	return lhs->cost > rhs->cost;
}