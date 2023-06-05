#include "pch.h"
#include "MinHeap.h"
#include <algorithm>

MinHeap::MinHeap(void)
{
	memset(arr, 0, sizeof(arr));
}

MinHeap::~MinHeap(void)
{

}

void MinHeap::Insert(Node* node)
{
	*(arr + heapSize++) = node;
	size_t i{ heapSize - 1 }, p = parent(i);
	while ( i > 0 && arr[p]->fx > arr[i]->fx )
	{
		Node* tmp = arr[i];
		arr[i] = arr[p];
		arr[p] = tmp;
		i = p; p = parent(i);
	}
}

void MinHeap::Rearrange(unsigned short id)
{
	size_t index{ std::numeric_limits<size_t>::max() };
	for (size_t i{}; i < heapSize; ++i)
	{
		if (arr[i]->info.id != id)
			continue;
		index = i;
		break;
	}

	size_t i{ index };
	while ( i > 0 && arr[parent(i)]->fx > arr[i]->fx )
	{
		std::swap( *(arr + i), *( arr + parent(i) ) );
		i = parent(i);
	}
}

Node* MinHeap::Pop(void)
{
	Node* min = *arr;
	*arr = *(arr + heapSize - 1);
	--heapSize;
	Heapify(0);

	return min;
}

void MinHeap::Clear(void)
{
	memset(arr, 0, sizeof(arr));
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
	for (size_t smallest{ i }; i < heapSize;)
	{
		size_t l = (i << 1) + 1;
		size_t r = (i << 1) + 2;

		if (l < heapSize && arr[l]->fx < arr[i]->fx)
			smallest = l;
		if (r < heapSize && arr[r]->fx < arr[smallest]->fx)
			smallest = r;

		if (smallest == i)
			break;
		Node* tmp = arr[i];
		arr[i] = arr[smallest];
		arr[smallest] = tmp;
		i = smallest;
	}
}

std::ostream& operator<<(std::ostream& os, MinHeap const& p)
{
	size_t const size = p.heapSize;
	//for (size_t i{}; i < size; ++i)
	//	os << "id: " << (*(p.arr + i))->info.id << " fx: " << (*(p.arr + i))->fx << std::endl;

	for (size_t i{}; i < size; ++i)
		os << " fx: " << (*(p.arr + i))->fx << std::endl;

	//for (size_t i{}; i < MAX_SIZE; ++i)
	//	std::cout << *(p.arr + i) << std::endl;

	return os << std::endl;
}