#ifndef	ICOLLECTOR_H
#define ICOLLECTOR_H

namespace ManCong
{
	namespace Memory
	{
		class ICollector
		{
		protected:
			ICollector(void);
			virtual ~ICollector(void) = 0;

			friend class CollectorManager;
			ICollector(ICollector const&) = delete;
			ICollector& operator=(ICollector const&) = delete;
			ICollector(ICollector&&) = delete;
			ICollector& operator=(ICollector&&) = delete;

			void free_memory(void);
		};
	}
}

#endif