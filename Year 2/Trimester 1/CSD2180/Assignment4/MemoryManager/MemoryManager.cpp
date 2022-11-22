#include "MemoryManager.h"

MemoryManager::MemoryManager(int total_bytes)
{
	mem = reinterpret_cast<char*>( malloc(static_cast<size_t>(total_bytes)) );
}

MemoryManager::~MemoryManager(void)
{
	free(mem);
}

void *MemoryManager::allocate(int bytes)
{

}

void MemoryManager::deallocate(void *ptr)
{

}

void MemoryManager::dump(std::ostream &out)
{

}
