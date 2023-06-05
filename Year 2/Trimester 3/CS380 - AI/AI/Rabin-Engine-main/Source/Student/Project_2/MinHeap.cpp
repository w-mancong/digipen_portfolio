#include "pch.h"
#include "MinHeap.h"
#include <algorithm>

MinHeap::MinHeap(void)
{
	arr = new Node * [MAX_WIDTH * MAX_HEIGHT];
	memset(arr, 0, sizeof(Node**) * MAX_WIDTH * MAX_HEIGHT);
}

MinHeap::~MinHeap(void)
{
	delete[] arr;
}

void MinHeap::Insert(Node* node)
{
	*(arr + heapSize++) = node;
}

void MinHeap::Heapify(void)
{
	for (int64_t i = ((heapSize - 1) >> 1); i >= 0; --i)
		Heapify(i);
}

Node* MinHeap::Pop(void)
{
	Heapify();
	Node* min = *arr;
	*arr = *(arr + heapSize - 1);
	--heapSize;
	Heapify(0);
	return min;
}

void MinHeap::Clear(void)
{
	for (size_t i{}; i < heapSize; ++i)
		*(arr + i) = nullptr;
	heapSize = 0;
}

bool MinHeap::Empty(void) const
{
	return !heapSize;
}

size_t MinHeap::size() const
{
	return heapSize;
}

size_t MinHeap::parent(size_t i) const
{
	return (i - 1) >> 1;
}

size_t MinHeap::left(size_t i) const
{
	return (i << 1) + 1;
}

size_t MinHeap::right(size_t i) const
{
	return (i << 1) + 2;
}

void MinHeap::Heapify(size_t i)
{
	size_t l = left(i);
	size_t r = right(i);

	size_t largest{ i };
	if ( l < heapSize && *(*(arr + l)) < *(*(arr + i)) )
		largest = l;

	if ( r < heapSize && *(*(arr + r)) < *(*(arr + largest)) )
		largest = r;

	if (largest != i)
	{
		std::swap( *(*(arr + i)), *(*(arr + largest)) );
		Heapify(largest);
	}
}

std::ostream& operator<<(std::ostream& os, MinHeap const& p)
{
	size_t const size = p.heapSize;
	for (size_t i{}; i < size; ++i)
		os << "id: " << (*(p.arr + i))->info.id << " fx: " << (*(p.arr + i))->finalCost << ' ';
	return os << std::endl;
}