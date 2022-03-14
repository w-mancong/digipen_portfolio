#include "ICollector.h"
#include "CollectorManager.h"

namespace ManCong
{
	ICollector::ICollector(void)
	{
		CollectorManager::AddCollection(this);
	};

	ICollector::~ICollector(void) {};

	void ICollector::free_memory(void) {};
}