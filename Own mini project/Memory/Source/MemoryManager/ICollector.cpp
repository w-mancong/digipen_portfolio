#include "pch.h"
#include "MemoryManager/ICollector.h"
#include "MemoryManager/CollectorManager.h"

namespace ManCong
{
	namespace Memory
	{
		ICollector::ICollector(void)
		{
			CollectorManager::AddCollection(this);
		};

		ICollector::~ICollector(void) {};

		void ICollector::free_memory(void) {};
	}
}