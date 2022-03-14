#include "ICollector.h"
#include "CollectorManager.h"

ICollector::ICollector(void)
{
	CollectorManager::AddCollection(this);
};

ICollector::~ICollector(void) {};

void ICollector::free_memory(void) {};