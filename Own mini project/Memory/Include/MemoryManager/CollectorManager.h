#ifndef	COLLECTOR_MANAGER_H
#define COLLECTOR_MANAGER_H

#include "pch.h"
namespace ManCong
{
	namespace Memory
	{
		class ICollector;
		class CollectorManager
		{
		public:
			static void AddCollection(ICollector* collection);
			static void FreeAll(void);

		private:
			CollectorManager(void);
			~CollectorManager(void);

			CollectorManager(CollectorManager const&) = delete;
			CollectorManager& operator=(CollectorManager const&) = delete;
			CollectorManager(CollectorManager&&) = delete;
			CollectorManager& operator=(CollectorManager&&) = delete;

			static std::vector<ICollector*> collections;
		};
	}
}

#endif