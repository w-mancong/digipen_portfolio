#include <iostream>
#include <list>
#include <iomanip>
#include <cstdlib>
#include <cstdio>

class MemoryManager
{
public:
    MemoryManager(int total_bytes);
    ~MemoryManager(void);
    void *allocate(int bytes);
    void deallocate(void *ptr);
    void dump(std::ostream &out);
};