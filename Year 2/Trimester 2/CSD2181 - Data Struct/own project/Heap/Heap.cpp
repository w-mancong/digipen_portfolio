#include "Heap.h"

namespace DataStruct
{
	MaxHeap::MaxHeap(void) : heapSize{ 0 }, cap{ 1 }, arr{ new value_type[cap] }
	{

	}

	MaxHeap::MaxHeap(pointer beg, pointer end)
	{
		cap = heapSize = end - beg;
		arr = new value_type[heapSize];
		std::copy(beg, end, arr);
		Build();
	}

	MaxHeap::~MaxHeap(void)
	{
		delete[] arr;
		arr = nullptr;
	}

	void MaxHeap::Build(void)
	{
		for (int64_t i = ((heapSize - 1) >> 1); i >= 0; --i)
			Heapify(i);
	}

	void MaxHeap::Sort(void)
	{
		Build();
		size_t size = heapSize;
		for (size_t i{ heapSize - 1 }; i >= 1; --i)
		{
			std::swap(*arr, *(arr + i));
			--heapSize;
			Heapify(0);
		}
		heapSize = size;
	}

	typename MaxHeap::value_type MaxHeap::Max(void)
	{
		return *arr;
	}

	typename MaxHeap::value_type MaxHeap::ExtractMax(void)
	{
		assert(heapSize > 0 && "heap underflow");
		Build();
		value_type max = Max();
		*arr = *(arr + heapSize - 1);
		--heapSize;
		Heapify(0);
		return max;
	}

	void MaxHeap::IncreaseKey(size_t i, value_type key)
	{
		assert(key > *(arr + i) && "new key is smaller than the current key");
		std::swap(*(arr + i), key);
		while (i > 0 && *(arr + parent(i)) < *(arr + i))
		{
			std::swap(*(arr + i), *(arr + parent(i)));
			i = parent(i);
		}
	}

	void MaxHeap::Insert(value_type key)
	{
		++heapSize;
		IncreaseCapacity();
		*(arr + heapSize - 1) = std::numeric_limits<value_type>::min();
		IncreaseKey(heapSize - 1, key);
	}

	typename MaxHeap::pointer MaxHeap::begin(void)
	{
		return const_cast<pointer>(const_cast<MaxHeap const&>(*this).begin());
	}

	typename MaxHeap::pointer MaxHeap::end(void)
	{
		return const_cast<pointer>(const_cast<MaxHeap const&>(*this).end());
	}

	typename MaxHeap::const_pointer MaxHeap::begin(void) const
	{
		return cbegin();
	}

	typename MaxHeap::const_pointer MaxHeap::end(void) const
	{
		return cend();
	}

	typename MaxHeap::const_pointer MaxHeap::cbegin(void) const
	{
		return arr;
	}

	typename MaxHeap::const_pointer MaxHeap::cend(void) const
	{
		return arr + heapSize;
	}

	size_t MaxHeap::size(void) const
	{
		return heapSize;
	}

	size_t MaxHeap::capacity(void) const
	{
		return cap;
	}

	s64 MaxHeap::parent(size_t i)
	{
		return (i - 1) >> 1;
	}

	s64 MaxHeap::left(size_t i)
	{
		return (i << 1) + 1;
	}

	s64 MaxHeap::right(size_t i)
	{
		return (i << 1) + 2;
	}

	void MaxHeap::Heapify(size_t i)
	{
		size_t l = left(i);
		size_t r = right(i);

		size_t largest{ i };
		if (l < heapSize && *(arr + l) > *(arr + i))
			largest = l;

		if (r < heapSize && *(arr + r) > *(arr + largest))
			largest = r;

		if (largest != i)
		{
			std::swap(*(arr + i), *(arr + largest));
			Heapify(largest);
		}
	}

	void MaxHeap::IncreaseCapacity(void)
	{
		if (heapSize <= cap)
			return;
		cap <<= 1; // increase by two times
		pointer tmp{ arr };
		arr = new value_type[cap];
		std::copy(tmp, tmp + heapSize - 1, arr);
		delete[] tmp;
		tmp = nullptr;
	}

	std::ostream& operator<<(std::ostream& os, MaxHeap const& p)
	{
		size_t const size = p.size();
		for (size_t i{}; i < size; ++i)
			os << *(p.begin() + i) << ' ';
		return os << std::endl;
	}
}