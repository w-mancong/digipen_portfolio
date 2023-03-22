/*!
file:	ALGraph.cpp
author:	Wong Man Cong
email:	w.mancong\@digipen.edu
brief:	This file contain function declaration for creating a graph data structure, and
		using Dijkstra's algorithm to find the shortest path from a single starting node

		All content © 2023 DigiPen Institute of Technology Singapore. All rights reserved.
*//*__________________________________________________________________________________*/
#include "ALGraph.h"

ALGraph::ALGraph(unsigned size)
{
	list.resize(static_cast<size_t>(size));
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
	MinHeap Q;
	std::vector<AdjInfo> nodes( list.size() );
	for (size_t i{}; i < nodes.size(); ++i)
	{
		unsigned id = static_cast<unsigned>(i) + 1;
		nodes[i] = { id, INF, 0 };
		Q.emplace_back( &nodes[i] );
	}

	nodes[start_node - 1].cost = 0;
	// heapify the vector
	std::make_heap( Q.begin(), Q.end(), AdjInfo() );

 	while (!Q.empty())
	{
		AdjInfo* u = Q.front();
		for (AdjacencyInfo const& v : list[u->id - 1])
		{
			// casting these to 8 byte unsigned int as u->cost + v.weight might overflow
			if ( static_cast<size_t>(nodes[v.id - 1].cost) > static_cast<size_t>(u->cost) + static_cast<size_t>(v.weight) )
			{
				nodes[v.id - 1].cost = u->cost + v.weight;
				nodes[v.id - 1].prev = u->id;
			}
		}
		// To decrease the key
		std::pop_heap ( Q.begin(), Q.end(), AdjInfo() );
		Q.pop_back();
		// Reheapify the vector
		std::make_heap( Q.begin(), Q.end(), AdjInfo() );
	}
	
	// Copying the data into DijkstraInfo
	std::vector<DijkstraInfo> res( list.size() );
	for (size_t i{}; i < res.size(); ++i)
	{
		res[i].cost = nodes[i].cost;
		unsigned id = nodes[i].id;
		if (res[i].cost == INF)	// if cost == INF means this node is isolated and other nodes are not connected to it
			continue;
		while (id != start_node && id != 0)
		{	// Retracking the path it took from my current id to start_node id
			res[i].path.emplace_back( nodes[id - 1].id );
			id = nodes[id - 1].prev;
		}
		res[i].path.emplace_back( start_node );
		std::reverse( res[i].path.begin(), res[i].path.end() );
	}

	return res;
}

ALIST const& ALGraph::GetAList(void) const
{
	return list;
}

bool ALGraph::AdjInfo::operator()(AdjInfo const* lhs, AdjInfo const* rhs) const
{
	if (lhs->cost == rhs->cost)
		return lhs->id > rhs->id;
	return lhs->cost > rhs->cost;
}