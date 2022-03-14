#include <iostream>
#include "MemoryManager.h"
//#include <cassert>
// Use (void) to silence unused warnings.
//#define assertm(exp, msg) assert(((void)msg, exp))

int main(void)
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	ManCong::MemoryManager<int>::GetInstance()->AllocateMemory(10);
	ManCong::CollectorManager::FreeAll();
}

//assert(2 + 2 == 4);
//std::cout << "Execution continues past the first assert\n";
//assertm(2 + 2 == 5, "There are five lights");
//std::cout << "Execution continues past the second assert\n";
//assert((2 * 2 == 4) && "Yet another way to add assert message");