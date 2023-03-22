/*!
file:	ALGraph.h
author:	Wong Man Cong
email:	w.mancong\@digipen.edu
brief:	This file contain function declaration for creating a graph data structure, and 
		using Dijkstra's algorithm to find the shortest path from a single starting node

		All content © 2023 DigiPen Institute of Technology Singapore. All rights reserved.
*//*__________________________________________________________________________________*/
//---------------------------------------------------------------------------
#ifndef ALGRAPH_H
#define ALGRAPH_H
//---------------------------------------------------------------------------
#include <vector>
#include <algorithm>

/*!*********************************************************************************
	\brief Struct used for containing the cost it takes to a specific node from
		   a starting node and the path to reach it
***********************************************************************************/
struct DijkstraInfo
{
	unsigned cost;
	std::vector<unsigned> path;
};

/*!*********************************************************************************
	\brief Struct for storing the the id of the node this current node can go to,
		   and the weight (distance) it cost to reach node id
***********************************************************************************/
struct AdjacencyInfo
{
	unsigned id;
	unsigned weight;
};

using ALIST = std::vector< std::vector<AdjacencyInfo> >;

class ALGraph
{
public:
	/*!*********************************************************************************
		\brief Default constructor for ALGraph

		\param [in] size: Total node that will exist in the graph
	***********************************************************************************/
	ALGraph(unsigned size);

	/*!*********************************************************************************
		\brief Default deconstructor
	***********************************************************************************/
	~ALGraph(void) = default;

	/*!*********************************************************************************
		\brief Add a directed edge from source to destination

		\param [in] source: Starting node
		\param [in] destination: Ending node
		\param [in] weight: Cost (distance) it takes to travel from source to destination
	***********************************************************************************/
	void AddDEdge(unsigned source, unsigned destination, unsigned weight);

	/*!*********************************************************************************
		\brief Add a undirected edge from source to destination

		\param [in] source: Starting node
		\param [in] destination: Ending node
		\param [in] weight: Cost (distance) it takes to travel from source to destination
	***********************************************************************************/
	void AddUEdge(unsigned node1, unsigned node2, unsigned weight);

	/*!*********************************************************************************
		\brief Runs Dijkstra's algorithm to find the shortest path to reach all the nodes
			   in the graph from a starting point
			  
		\param [in] start_node: ID of node that to start the algorithm from

		\return A vector of DijkstraInfo containing the details of the cost and path
				to take to reach the specific node from start_node
	***********************************************************************************/
	std::vector<DijkstraInfo> Dijkstra(unsigned start_node) const;

	/*!*********************************************************************************
		\brief Return a reference to the list of adjacency info of the graph
	***********************************************************************************/
	ALIST const& GetAList(void) const;

private:
	static constexpr const unsigned INF = std::numeric_limits<unsigned>::max();

	/*!*********************************************************************************
		\brief Used to store the relevant data for computing the Dijkstra's algorithm
	***********************************************************************************/
	struct AdjInfo
	{
		/*!*********************************************************************************
			\brief Functor for comparing two pointer to AdjInfo by cost, or if the cost
				   is the same between the lhs and rhs, compare using the id
		***********************************************************************************/
		bool operator()(AdjInfo const* lhs, AdjInfo const* rhs) const;

		/*
			id: ID of the current node
			cost: Cost it takes to reach this current node from start node
			prev: Prev node id that reaches to this node from the cost
		*/
		unsigned id{ 0 }, cost{ 0 }, prev{ 0 };
	};

	using MinHeap = std::vector<AdjInfo*>;
	ALIST list{};	// list containing the graph data
};

#endif
