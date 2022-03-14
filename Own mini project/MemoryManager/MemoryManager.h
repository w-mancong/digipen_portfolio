#ifndef	MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include "MemoryLeak.h"
#include "Helper.h"
#include "CollectorManager.h"
#include "ICollector.h"

template <class T>
class MemoryManager : public ICollector
{
public:
	static MemoryManager<T>* GetInstance(void)
	{
		if (instance)
			return instance;
		return (instance = new MemoryManager<T>);
	}

	T* AllocateMemory(size_t size = 1)
	{
		memory.push_back(new T[size]{});
		return memory.back();
	}

	T* AllocateMemory(size_t width, size_t height)
	{
		memory.push_back(new T[width * height]);
		return memory.back();
	}

private:
	MemoryManager(void) {}
	~MemoryManager(void)
	{
		while (memory.size())
		{
			if (memory.back())
			{
				delete[] memory.back();
				memory.back() = nullptr;
				memory.pop_back();
			}
		}
	}

	MemoryManager(MemoryManager const&) = delete;
	MemoryManager& operator=(MemoryManager const&) = delete;
	MemoryManager(MemoryManager&&) = delete;
	MemoryManager& operator=(MemoryManager&&) = delete;

	static MemoryManager<T>* instance;
	std::vector<T*> memory;
};

template <class T>
MemoryManager<T>* MemoryManager<T>::instance = nullptr;

#endif