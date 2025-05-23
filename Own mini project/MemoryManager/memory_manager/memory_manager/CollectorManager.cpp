#include "CollectorManager.h"
#include "ICollector.h"

extern char const* bad_allocations;
namespace ManCong
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
		if (bad_allocations)
		{
			delete[] bad_allocations;
			bad_allocations = nullptr;
		}
	}
}