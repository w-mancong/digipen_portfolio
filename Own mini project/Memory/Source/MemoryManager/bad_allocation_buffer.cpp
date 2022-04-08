#include "pch.h"
// Remember to call CollectorManager::FreeAll() to avoid any leaks in memory
// Reserve memory so that if there is a bad allocation, there is enough memory to print out error message
char const* bad_allocations_buffer = new char[16384];