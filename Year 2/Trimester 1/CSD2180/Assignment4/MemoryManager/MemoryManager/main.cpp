#include "MemoryManager.h"

int main(void)
{
	MemoryManager mm(128);
	mm.allocate(4);
	void* ptr = mm.allocate(8);
	mm.allocate(4);

	mm.deallocate(ptr);

	mm.dump(); std::cout << std::endl;

	ptr = mm.allocate(3);
	void* ptr2 = mm.allocate(8);
	mm.dump(); std::cout << std::endl;

	mm.deallocate(ptr);
	mm.deallocate(ptr2);
	mm.dump(); std::cout << std::endl;
}
