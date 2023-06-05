#pragma once
#include "Node.h"

class OpenList
{
public:
	OpenList();
	~OpenList();
	void Insert(Node* node);
	
private:
	static size_t constexpr const MAX_SUB_BUCKETS{ 5 }, MAX_NODES_IN_SUB_BUCKET{ 64 };
	static float constexpr const DIVISION{ 1.0f / static_cast<float>(MAX_SUB_BUCKETS) };

	size_t totalNodes{}, smallestIndex{};
	std::vector< std::vector< std::vector<Node*> > > list{};
};