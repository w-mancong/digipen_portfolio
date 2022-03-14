#ifndef	MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include "MemoryLeak.h"
#include "Helper.h"
#include "CollectorManager.h"
#include "ICollector.h"
#include "VariableType.h"

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
		return Allocate(size);
	}

	T* AllocateMemory(size_t width, size_t height)
	{
		return Allocate(width * height);
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

	T* Allocate(size_t size)
	{
		memory.push_back(new T[size]{});
		return memory.back();
	}

	static MemoryManager<T>* instance;
	std::vector<T*> memory;
};

template <class T>
MemoryManager<T>* MemoryManager<T>::instance = nullptr;

//std::cout << "decltype is " << type_name<decltype(*this)>() << std::endl;

#endif