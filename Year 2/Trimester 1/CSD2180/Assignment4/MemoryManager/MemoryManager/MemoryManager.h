
#include <iostream>
#include <list>
#include <iomanip>
#include <cstdlib>

class MemoryManager
{
public:
    MemoryManager(uint64_t total_bytes);
    ~MemoryManager(void);
    void *allocate(uint64_t bytes);
    void deallocate(void *ptr);
    void dump(std::ostream &out = std::cout);

private:
    struct Node
    {
        char* startAddress{ nullptr };
        uint64_t byteCount{ 0 };
        bool allocated{ false };
    };

    void Print(void* startAddress, uint64_t byteCount, bool allocated, std::ostream& out);

    uint64_t m_TotalBytes{ 0 }, m_AllocatedBytes{ 0 };
    char* m_Memory{ nullptr };
    std::list<Node> m_NodeList{};
};