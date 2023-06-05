#include <pch.h>
#include "OpenList.h"

OpenList::OpenList() : totalNodes{ 0 }, smallestIndex{ std::numeric_limits<size_t>::max() }
{
	list.resize(MAX_SIZE << 1);
	for (size_t i{}; i < MAX_SIZE << 1; ++i)
	{
		list[i].resize(MAX_SUB_BUCKETS);
		for (size_t j{}; j < MAX_SUB_BUCKETS; ++j)
			list[i][j].resize(MAX_NODES_IN_SUB_BUCKET);
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