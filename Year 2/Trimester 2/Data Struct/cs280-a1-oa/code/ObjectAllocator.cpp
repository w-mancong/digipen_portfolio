#include <cstring>
#include "ObjectAllocator.h"

namespace
{
    size_t constexpr operator "" _z(size_t n)
    {
        return static_cast<size_t>(n);
    }
}

// Creates the ObjectManager per the specified values
// Throws an exception if the construction fails. (Memory allocation problem)
ObjectAllocator::ObjectAllocator(size_t ObjectSize, const OAConfig &config) : Config_(config)
{
    Stats_.ObjectSize_ = ObjectSize;
    /*
        Page size consist of:
        Pointer to the next block, left alignment, middle block
        Middle block -> Header block, Padding, Size of Object, Padding, Inter alignment
    */
    middleBlockSize = Config_.HBlockInfo_.size_ + (Config_.PadBytes_ * 2_z) + Stats_.ObjectSize_ + Config_.InterAlignSize_;
    Stats_.PageSize_ = sizeof(void*) + Config_.LeftAlignSize_ + (middleBlockSize * Config_.ObjectsPerPage_) - Config_.InterAlignSize_;
   
    AllocateNewPage();
}

// Destroys the ObjectManager (never throws)
ObjectAllocator::~ObjectAllocator()
{

}

// Take an object from the free list and give it to the client (simulates new)
// Throws an exception if the object can't be allocated. (Memory allocation problem)
void *ObjectAllocator::Allocate(const char *label)
{
    if(Config_.UseCPPMemManager_)
    {
        try
        {
            unsigned char *ptr = new unsigned char[Stats_.ObjectSize_];
            // increment value
            return ptr;
        }
        catch(std::bad_alloc const&)
        {
            throw OAException(OAException::E_NO_MEMORY, "Out of memory!");
        }        
    }
}

// Returns an object to the free list for the client (simulates delete)
// Throws an exception if the the object can't be freed. (Invalid object)
void ObjectAllocator::Free(void *Object)
{

}

// Calls the callback fn for each block still in use - returns the number of blocks in used by the client
unsigned ObjectAllocator::DumpMemoryInUse(DUMPCALLBACK fn) const
{
    return 0;
}

// Calls the callback fn for each block that is potentially corrupted
// The ValidatePages method returns the number of blocks that are corrupted. Only pad bytes are validated.
unsigned ObjectAllocator::ValidatePages(VALIDATECALLBACK fn) const
{
    return 0;
}

// Frees all empty page - returns the number of pages freed
unsigned ObjectAllocator::FreeEmptyPages()
{
    return 0;
}

// Testing/Debugging/Statistic methods
void ObjectAllocator::SetDebugState(bool State) // true=enable, false=disable
{
    Config_.DebugOn_ = State;
}  

const void *ObjectAllocator::GetFreeList() const // returns a pointer to the internal free list
{
    return FreeList_;
}

const void *ObjectAllocator::GetPageList() const // returns a pointer to the internal page list
{
    return PageList_;
}

OAConfig ObjectAllocator::GetConfig() const      // returns the configuration parameters
{
    return Config_;
}

OAStats ObjectAllocator::GetStats() const        // returns the statistics for the allocator
{
    return Stats_;
}

void ObjectAllocator::AllocateNewPage(void)
{
    if (totalNumberOfPages >= Config_.MaxPages_)
        return;

    // Allocate a new page and insert it at the front of the PageList_
    unsigned char *ptr = nullptr;
    try
    {
        ptr = new unsigned char[Stats_.PageSize_];
            
        // Allocating spaces for page
        if (!PageList_)
        {   // PageList_ is nullptr, so this very first page will be the head
            PageList_ = reinterpret_cast<GenericObject*>(ptr);
            PageList_->Next = nullptr;
        }
        else
        {
            GenericObject* head = reinterpret_cast<GenericObject*>(ptr);
            head->Next = PageList_;
            PageList_ = head;
        }
    }
    catch (std::bad_alloc const&)
    {
        throw OAException(OAException::E_NO_MEMORY, "No more memory!");
    }

    ++totalNumberOfPages;
    AssignFreeListObjects();
    AssignByteSignatures();
}

void ObjectAllocator::AssignFreeListObjects(void)
{
    unsigned char* head = reinterpret_cast<unsigned char*>( PageList_ );
    size_t const FRONT_OFFSET = sizeof(void*) + static_cast<size_t>(Config_.LeftAlignSize_) + static_cast<size_t>(Config_.HBlockInfo_.size_) + static_cast<size_t>(Config_.PadBytes_);
    FreeList_ = reinterpret_cast<GenericObject*>( head + FRONT_OFFSET );
    FreeList_->Next = nullptr;

    for (size_t i = 1; i < Config_.ObjectsPerPage_; ++i)
    {
        GenericObject* ptr = reinterpret_cast<GenericObject*>(head + FRONT_OFFSET + (middleBlockSize * i));
        ptr->Next = FreeList_;
        FreeList_ = ptr;
    }
}

void ObjectAllocator::AssignByteSignatures(void)
{
    // TODO: Set byte alignment for left align
    GenericObject* go = FreeList_;
    for (size_t i = 0; i < Config_.ObjectsPerPage_; ++i)
    {
        unsigned char* ptr = reinterpret_cast<unsigned char*>(go);

        // Setting the byte signatures
        // Padding
        memset(ptr + Config_.HBlockInfo_.size_ + sizeof(void*), PAD_PATTERN, static_cast<size_t>(Config_.PadBytes_));
        // Unallocated memory
        memset(ptr + Config_.HBlockInfo_.size_ + Config_.PadBytes_ + sizeof(void*), UNALLOCATED_PATTERN, static_cast<size_t>(Stats_.ObjectSize_) - sizeof(void*));
        // Padding
        memset(ptr + Config_.HBlockInfo_.size_ + Config_.PadBytes_ + sizeof(void*) + Stats_.ObjectSize_, PAD_PATTERN, static_cast<size_t>(Config_.PadBytes_));
        // Interalignment
        memset(ptr + Config_.HBlockInfo_.size_ + (Config_.PadBytes_ * 2_z) + sizeof(void*) + Stats_.ObjectSize_, ALIGN_PATTERN, static_cast<size_t>(Config_.InterAlignSize_));

        go = go->Next;
    }
}
