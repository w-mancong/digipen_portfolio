#include <cstring>
#include "ObjectAllocator.h"

namespace
{
    size_t constexpr operator "" _z(size_t n)
    {
        return static_cast<size_t>(n);
    }

    inline size_t AlignByte(size_t n, size_t align)
    {
        if (!align) return n;
        size_t r = n % align != 0 ? 1_z : 0_z;  // Getting the remainer
        return align * ((n / align) + r);
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
    leftAlignSize = sizeof(void*) + Config_.HBlockInfo_.size_ + Config_.PadBytes_;
    dataBlockSize = Stats_.ObjectSize_ + (Config_.PadBytes_ * 2_z) + Config_.HBlockInfo_.size_;

    size_t const LEFT_ALIGNMENT_OFFSET  = AlignByte(leftAlignSize,  static_cast<size_t>(Config_.Alignment_)),
                 INTER_ALIGNMENT_OFFSET = AlignByte(dataBlockSize, static_cast<size_t>(Config_.Alignment_));

    Config_.LeftAlignSize_  = LEFT_ALIGNMENT_OFFSET  - leftAlignSize;
    Config_.InterAlignSize_ = INTER_ALIGNMENT_OFFSET - dataBlockSize;
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
            IncrementStatsValue();
            unsigned char *ptr = new unsigned char[Stats_.ObjectSize_];
            return ptr;
        }
        catch(std::bad_alloc const&)
        {
            throw OAException(OAException::E_NO_MEMORY, "Out of memory!");
        }        
    }

    // Check if FreeList_ have any objects left
    if (!FreeList_)
    {
        // If no more free objects in FreeList_ left, AllocateNewPage
        AllocateNewPage();
    }

    // Temp ptr to store the pointer to the object to be given to client
    void* ptr = reinterpret_cast<void*>(FreeList_);
    FreeList_ = FreeList_->Next;

    IncrementStatsValue();

    // Update the byte signature
    UpdateByteSignature(reinterpret_cast<unsigned char*>(ptr), ALLOCATED_PATTERN, Stats_.ObjectSize_);

    // Update the header block based on the type it is

    return ptr;
}

// Returns an object to the free list for the client (simulates delete)
// Throws an exception if the the object can't be freed. (Invalid object)
void ObjectAllocator::Free(void *Object)
{
    ++Stats_.Deallocations_;

    if (Config_.UseCPPMemManager_)
    {
        delete[] reinterpret_cast<unsigned char*>(Object);
        return;
    }

    GenericObject* ptr = reinterpret_cast<GenericObject*>(Object);
    ptr->Next = nullptr;

    // Check if Object is within the memory boundary and if the data was corrupted in any way
    if (Config_.DebugOn_)
    {
        if ( !WithinMemoryBoundary( reinterpret_cast<unsigned char*>(ptr) ) )
            throw OAException(OAException::E_BAD_BOUNDARY, "Memory not within boundary!");
        if ( IsPaddingCorrupted( GetPadding(ptr, Padding::Left) ) || IsPaddingCorrupted( GetPadding(ptr, Padding::Right) ) )
            throw OAException(OAException::E_CORRUPTED_BLOCK, "Corrupted memory!");
    }

    --Stats_.ObjectsInUse_;

    AddObjectToFreeList(ptr);
    UpdateByteSignature(reinterpret_cast<unsigned char*>(ptr), FREED_PATTERN, Stats_.ObjectSize_);

    // Free the header block based on the type it is
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

void ObjectAllocator::IncrementStatsValue(void)
{
    --Stats_.FreeObjects_, ++Stats_.Allocations_, ++Stats_.ObjectsInUse_;
    if (Stats_.MostObjects_ < Stats_.ObjectsInUse_)
        Stats_.MostObjects_ = Stats_.ObjectsInUse_;
}

void ObjectAllocator::AllocateNewPage(void)
{
    if (Stats_.PagesInUse_ >= Config_.MaxPages_)
        throw OAException(OAException::E_NO_PAGES, "Out of Pages!");

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

    ++Stats_.PagesInUse_;

    AssignFreeListObjects();
    AssignByteSignatures();
    DefaultBlockValue();
}

void ObjectAllocator::AssignFreeListObjects(void)
{
    unsigned char* head = reinterpret_cast<unsigned char*>( PageList_ );
    size_t const FRONT_OFFSET = sizeof(void*) + static_cast<size_t>(Config_.LeftAlignSize_) + static_cast<size_t>(Config_.HBlockInfo_.size_) + static_cast<size_t>(Config_.PadBytes_);
    FreeList_ = reinterpret_cast<GenericObject*>( head + FRONT_OFFSET );
    FreeList_->Next = nullptr;
    ++Stats_.FreeObjects_;

    for (size_t i = 1; i < Config_.ObjectsPerPage_; ++i)
    {
        GenericObject* ptr = reinterpret_cast<GenericObject*>(head + FRONT_OFFSET + (middleBlockSize * i));
        AddObjectToFreeList(ptr);
    }
}

void ObjectAllocator::AssignByteSignatures(void)
{
    if (!Config_.DebugOn_)
        return;

    // TODO: Set byte alignment for left align
    GenericObject* go = PageList_;
    unsigned char* ptr = reinterpret_cast<unsigned char*>(go);

    // Left alignment
    memset(ptr + sizeof(void*), ALIGN_PATTERN, Config_.LeftAlignSize_);

    for (size_t i = 0; i < Config_.ObjectsPerPage_; ++i)
    {
        // Setting the byte signatures
        // Padding
        size_t OFFSET = sizeof(void*) + Config_.LeftAlignSize_ + Config_.HBlockInfo_.size_ + (middleBlockSize * i);
        memset(ptr + OFFSET, PAD_PATTERN, static_cast<size_t>(Config_.PadBytes_));

        // Unallocated memory
        OFFSET += Config_.PadBytes_ + sizeof(void*);
        memset(ptr + OFFSET, UNALLOCATED_PATTERN, static_cast<size_t>(Stats_.ObjectSize_ - sizeof(void*)));

        // Padding
        OFFSET += Stats_.ObjectSize_ - sizeof(void*);
        memset(ptr + OFFSET, PAD_PATTERN, static_cast<size_t>(Config_.PadBytes_));

        // Don't do the signature for the page when it reaches the end
        if (i + 1 >= Config_.ObjectsPerPage_) break;

        // Interalignment
        OFFSET += Config_.PadBytes_;
        memset(ptr + OFFSET, ALIGN_PATTERN, Config_.InterAlignSize_);
    }
}

void ObjectAllocator::DefaultBlockValue(void)
{
    GenericObject* go = FreeList_;

    switch (Config_.HBlockInfo_.type_)
    {
        case OAConfig::HBLOCK_TYPE::hbBasic:
        case OAConfig::HBLOCK_TYPE::hbExtended:
        {
            while (go)
            {
                unsigned char* ptr = GetHeaderAddress(go);
                memset(ptr, 0, Config_.HBlockInfo_.size_);
                go = go->Next;
            }
            break;
        }
        case OAConfig::HBLOCK_TYPE::hbExternal:
        {
            while (go)
            {
                GenericObject** ptr = reinterpret_cast<GenericObject**>( GetHeaderAddress(go) );
                *ptr = nullptr; 
                go = go->Next;
            }

            break;
        }
    }
}

void ObjectAllocator::AddObjectToFreeList(GenericObject* obj)
{
    GenericObject* ptr = FreeList_;
    FreeList_ = obj;
    obj->Next = ptr;
    ++Stats_.FreeObjects_;
}

unsigned char* ObjectAllocator::GetHeaderAddress(void* ptr)
{
    return reinterpret_cast<unsigned char*>(ptr) - Config_.PadBytes_ - Config_.HBlockInfo_.size_;
}

void ObjectAllocator::UpdateByteSignature(unsigned char* ptr, unsigned char c, size_t size) const
{
    if(Config_.DebugOn_)
        memset(ptr, c, size);
}

bool ObjectAllocator::WithinMemoryBoundary(unsigned char* ptr) const
{
    /*
        1) Check if ptr is within the confines of all the pages available
        2) Check if ptr is an address before the first block of object
        3) Check if ptr is divisible by the data block size
    */

    if (!WithinPage(ptr)) // 1) Check if ptr is within the confines of all the pages available
        throw OAException(OAException::E_BAD_BOUNDARY, "Pointer out of boundary!");

    return false;
}

unsigned char* ObjectAllocator::GetPadding(GenericObject* ptr, Padding p) const
{
    switch (p)
    {
        case Padding::Left : return reinterpret_cast<unsigned char*>(ptr) - Config_.PadBytes_;
        case Padding::Right: return reinterpret_cast<unsigned char*>(ptr) + Stats_.ObjectSize_;
        default: return nullptr;
    }
}

bool ObjectAllocator::IsPaddingCorrupted(unsigned char* ptr) const
{
    // Anything other than PAD_PATTERN would means that the padding is corrupted
    for (size_t i = 0; i < Config_.PadBytes_; ++i)
        if (*(ptr + i) != PAD_PATTERN) return true;
    return false;
}

bool ObjectAllocator::WithinPage(unsigned char* ptr) const
{
    GenericObject* page = PageList_;
    while (page)
    {
        unsigned char* pagePtr = reinterpret_cast<unsigned char*>(page);
        if (ptr >= pagePtr && ptr < pagePtr + Stats_.PageSize_) return true;
        page = page->Next;
    }
    return false;
}
