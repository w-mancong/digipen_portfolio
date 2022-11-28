#include "MemoryManager.h"

int main(void)
{
	MemoryManager mm(128);
	mm.allocate(4);
	void* ptr = mm.allocate(8);
	mm.allocate(4);

	//mm.deallocate(ptr);

	mm.allocate(110);
	mm.allocate(2);
	mm.dump();
	mm.allocate(2);

}
