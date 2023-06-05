#include <pch.h>
#include "OpenList.h"

OpenList::OpenList() : totalNodes{ 0 }, smallestIndex{ std::numeric_limits<size_t>::max() }
{
	size_t const LIST_SIZE = MAX_SIZE << 1;
	list.resize(LIST_SIZE);
	for (size_t i{}; i < LIST_SIZE; ++i)
	{
		list[i].resize(MAX_SUB_BUCKETS);
		for (size_t j{}; j < MAX_SUB_BUCKETS; ++j)
			list[i][j].reserve(MAX_NODES_IN_SUB_BUCKET);
	}
}

OpenList::~OpenList()
{

}

void OpenList::Insert(Node* node)
{
	float wholeValue = std::floor(node->fx),
		decimalValue = node->fx - wholeValue;

	size_t bucketIndex = static_cast<size_t>(wholeValue),
		subBucketIndex = static_cast<size_t>(decimalValue / DIVISION);

	list[bucketIndex][subBucketIndex].emplace_back(node);
	if (bucketIndex < smallestIndex)
		smallestIndex = bucketIndex;
	++totalNodes;
}

void OpenList::Remove(Node const* node)
{
	float wholeValue = std::floor(node->fx),
		decimalValue = node->fx - wholeValue;

	size_t bucketIndex = static_cast<size_t>(wholeValue),
		subBucketIndex = static_cast<size_t>(decimalValue / DIVISION);

	auto it = std::find_if(list[bucketIndex][subBucketIndex].begin(), list[bucketIndex][subBucketIndex].end(), [node](Node const* lhs)
		{
			return lhs->info.id == node->info.id;
		});
	list[bucketIndex][subBucketIndex].erase(it);
}

Node* OpenList::Pop(void)
{
	Node* node{ nullptr }; size_t totalNodesInThisSubBucket{ 0 };
	// Using this loop to search for the lowest sub bucket index
	for (size_t i{}; i < MAX_SUB_BUCKETS; ++i)
	{
		if ( !list[smallestIndex][i].size() || node )
		{
			totalNodesInThisSubBucket += list[smallestIndex][i].size();
			continue;
		}

		size_t smallestNode{ 0 };
		// Using this for loop to search for the lowest fx in this vector
		for (size_t j{ 1 }; j < list[smallestIndex][i].size(); ++j)
		{
			if ( list[smallestIndex][i][j]->fx > list[smallestIndex][i][smallestNode]->fx )
				continue;
			smallestNode = j;
		}

		// By the end of this loop, node will now contain the pointer to the lowest fx
		node = list[smallestIndex][i][smallestNode];
	}

	if (totalNodesInThisSubBucket <= 0)
	{
		size_t index = smallestIndex, LIST_SIZE = list.size();
		smallestIndex = std::numeric_limits<size_t>::max();

		for (size_t i{ index }; i < LIST_SIZE; ++i)
		{
			if (!list[i].size())
				continue;

			smallestIndex = i;
			break;
		}
	}

	--totalNodes;
	return node;
}

void OpenList::Clear(void)
{
	for (size_t i{}; i < MAX_SIZE << 1; ++i)
	{
		for (size_t j{}; j < MAX_SUB_BUCKETS; ++j)
			list[i][j].clear();
	}
	totalNodes = 0;
}

bool OpenList::Empty(void) const
{
	return !totalNodes;
}
