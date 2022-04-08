#include "pch.h"
#include "MemoryManager/CollectorManager.h"
#include "MemoryManager/ICollector.h"

extern char const* bad_allocations_buffer;
namespace ManCong
{
	namespace Memory
	{
		std::vector<ICollector*> CollectorManager::collections;

		CollectorManager::CollectorManager(void) {};
		CollectorManager::~CollectorManager(void) {};

		void CollectorManager::AddCollection(ICollector* collection)
		{
			collections.push_back(collection);
		}

		void CollectorManager::FreeAll(void)
		{
			while (collections.size())
			{
				if (collections.back())
				{
					collections.back()->free_memory();
					delete collections.back();
					collections.back() = nullptr;
					collections.pop_back();
				}
			}
			if (bad_allocations_buffer)
			{
				delete[] bad_allocations_buffer;
				bad_allocations_buffer = nullptr;
			}
		}
	}
}