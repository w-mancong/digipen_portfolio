#pragma once
#include "Node.h"

class QuickArray
{
public:
	void Insert(Node* node);
	Node* Pop(void);
	bool Empty(void) const;
	void Clear(void);

private:
	size_t totalNodes{};
	std::array<Node*, 160> arr;
};