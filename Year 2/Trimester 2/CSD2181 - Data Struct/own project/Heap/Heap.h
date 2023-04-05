#ifndef	HEAP_H
#define HEAP_H

#include <algorithm>
#include <cassert>
#include <iostream>

namespace DataStruct
{
	using u8  = uint8_t;
	using s8  = int8_t;
	using u16 = uint16_t;
	using s16 = int16_t;
	using u32 = uint32_t;
	using s32 = int32_t;
	using s64 = int64_t;
	using u64 = uint64_t;
	class MaxHeap
	{
	public:
		using value_type = int;
		using pointer = value_type*;
		using const_pointer = value_type const*;
		using reference = value_type&;
		using const_reference = value_type const&;

		MaxHeap(void);
		MaxHeap(pointer beg, pointer end);
		~MaxHeap(void);

		// O(n)
		void Build(void);
		// O(n log n)
		void Sort(void);
		// O(1)
		value_type Max(void);
		// O(log n)
		value_type ExtractMax(void);
		// O(log n)
		void IncreaseKey(size_t i, value_type key);
		// O(log n)
		void Insert(value_type key);

		pointer begin(void);
		pointer end(void);

		const_pointer begin(void) const;
		const_pointer end(void) const;

		const_pointer cbegin(void) const;
		const_pointer cend(void) const;

		size_t size(void) const;
		size_t capacity(void) const;

	private:
		s64 parent(size_t i);
		s64 left(size_t i);
		s64 right(size_t i);

		// O(log n)
		void Heapify(size_t i);
		void IncreaseCapacity(void);

		size_t heapSize{}, cap{};
		pointer arr{ nullptr };
	};

	std::ostream& operator<<(std::ostream& os, MaxHeap const& p);
}

#endif