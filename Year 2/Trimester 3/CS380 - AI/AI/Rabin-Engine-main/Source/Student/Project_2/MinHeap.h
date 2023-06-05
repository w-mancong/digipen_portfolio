#pragma once
#include "Node.h"

class MinHeap
{
public:
	MinHeap(void);
    ~MinHeap(void);

    void Insert(Node* node);
    void Rearrange(unsigned short id);
    Node* Pop(void);
    void Clear(void);
    bool Empty(void) const;
    size_t size() const;

private:
    size_t parent(size_t i) const;
    size_t left(size_t i) const;
    size_t right(size_t i) const;
    void Heapify(size_t i);

    Node* arr[MAX_SIZE];
    size_t heapSize{};

    friend std::ostream& operator<<(std::ostream& os, MinHeap const& p);
};

std::ostream& operator<<(std::ostream& os, MinHeap const& p);