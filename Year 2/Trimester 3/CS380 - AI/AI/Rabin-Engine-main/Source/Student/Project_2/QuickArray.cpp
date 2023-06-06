#include <pch.h>
#include "QuickArray.h"

void QuickArray::Insert(Node* node)
{
	arr[totalNodes++] = node;
}

Node* QuickArray::Pop(void)
{
	// Search thru the entire arr to find the smallest float value
	struct Lowest
	{
		size_t index{};
		float fx{ std::numeric_limits<float>::max() };
	} lowest;

	for (size_t i{}; i < totalNodes; ++i)
	{
		Node* n = arr[i];
		if (n->fx > lowest.fx)
			continue;
		lowest.index = i;
		lowest.fx = n->fx;
	}
	--totalNodes;
	return arr[lowest.index];
}

bool QuickArray::Empty(void) const
{
	return !totalNodes;
}

void QuickArray::Clear(void)
{
	arr.fill(nullptr);
	totalNodes = 0;
}
