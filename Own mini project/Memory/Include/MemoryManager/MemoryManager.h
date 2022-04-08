#ifndef	MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include "MemoryLeak.h"
#include "Helper.h"
#include "CollectorManager.h"
#include "ICollector.h"
#include "VariableType.h"

extern const char* bad_allocations_buffer;
namespace ManCong
{
	namespace Memory
	{
		template <class T>
		void Deallocate(T const*& ptr)
		{
			delete[] ptr;
			ptr = nullptr;
		}

		template <class T>
		class MemoryManager : private Memory::ICollector
		{
		public:
			static MemoryManager<T>* Get(void)
			{
				return instance ? instance : (instance = new MemoryManager<T>{});
			}

			T* New(size_t size = 1)
			{
				return Allocate(size);
			}

			T* New(size_t width, size_t height)
			{
				return Allocate(width * height);
			}

			void Delete(T const* ptr)
			{
				for (auto it = memory.cbegin(); it < memory.cend(); ++it)
				{
					if (*it != ptr)
						continue;
					memory.erase(it);
					Deallocate(ptr);
					break;
				}
			}

		private:
			MemoryManager(void) {}
			~MemoryManager(void)
			{
				while (memory.size())
				{
					if (memory.back())
					{
						Deallocate(memory.back());
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
				try
				{
					memory.push_back(new T[size]{});
				}
				catch (std::bad_alloc ex)
				{
					Deallocate(bad_allocations_buffer);
					std::cerr << "exception caught: " << ex.what() << std::endl;
					std::cerr << "Program Exit: (Allocating memory for " << type_name<decltype(this)>() << ")" << std::endl;
					CollectorManager::FreeAll();
					exit(EXIT_FAILURE);
				}
				return const_cast<T*>(memory.back());
			}

			static MemoryManager<T>* instance;
			std::vector<T const*> memory;
		};

		template <class T>
		MemoryManager<T>* MemoryManager<T>::instance = nullptr;
	}
}

#endif