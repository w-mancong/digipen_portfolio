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

    Config_.LeftAlignSize_  = static_cast<unsigned int>(LEFT_ALIGNMENT_OFFSET  - leftAlignSize);
    Config_.InterAlignSize_ = static_cast<unsigned int>(INTER_ALIGNMENT_OFFSET - dataBlockSize);
    middleBlockSize = Config_.HBlockInfo_.size_ + (Config_.PadBytes_ * 2_z) + Stats_.ObjectSize_ + Config_.InterAlignSize_;
    Stats_.PageSize_ = sizeof(void*) + Config_.LeftAlignSize_ + (middleBlockSize * Config_.ObjectsPerPage_) - Config_.InterAlignSize_;
   
    AllocateNewPage();
}

// Destroys the ObjectManager (never throws)
ObjectAllocator::~ObjectAllocator()
{
    GenericObject* ptr = PageList_;

    while (ptr)
    {
        GenericObject* n = ptr->Next;

        // Check if the header block type is of External
        if (Config_.HBlockInfo_.type_ == OAConfig::HBLOCK_TYPE::hbExternal)
        {   // Free the MemBlockInfo
            for (size_t i = 0; i < Config_.ObjectsPerPage_; ++i)
            {
                size_t const OFFSET = sizeof(void*) + Config_.LeftAlignSize_ + (middleBlockSize * i);
                MemBlockInfo** header = reinterpret_cast<MemBlockInfo**>(reinterpret_cast<unsigned char*>(ptr) + OFFSET);
                if (header)
                {
                    if ((*header))
                    {
                        delete[] (*header)->label;
                        (*header)->label = nullptr;
                    }

                    delete *header;
                    *header = nullptr;
                }
            }
        }

        delete[] ptr;
        ptr = n;
    }
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
    GenericObject* ptr = FreeList_;
    FreeList_ = FreeList_->Next;

    IncrementStatsValue();

    // Update the byte signature
    UpdateByteSignature(reinterpret_cast<unsigned char*>(ptr), ALLOCATED_PATTERN, Stats_.ObjectSize_);

    // Update the header block based on the type it is
    UpdateHeader(ptr, label);

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

    ReleaseHeader(ptr);
    AddObjectToFreeList(ptr);
    UpdateByteSignature(reinterpret_cast<unsigned char*>(ptr), FREED_PATTERN, Stats_.ObjectSize_);
}

// Calls the callback fn for each block still in use - returns the number of blocks in used by the client
unsigned ObjectAllocator::DumpMemoryInUse(DUMPCALLBACK fn) const
{
    // Empty Page
    if (!PageList_)
        return 0;

    unsigned int numBlockMemUsed = 0;

    GenericObject* page = PageList_;
    while (page)
    {

    }

    return numBlockMemUsed;
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

unsigned char* ObjectAllocator::GetHeaderAddress(void* ptr) const
{
    return reinterpret_cast<unsigned char*>(ptr) - Config_.PadBytes_ - Config_.HBlockInfo_.size_;
}

void ObjectAllocator::UpdateHeader(GenericObject* ptr, char const* label) const
{
    switch (Config_.HBlockInfo_.type_)
    {
        case OAConfig::HBLOCK_TYPE::hbBasic:
        {
            BasicBlockHeader(ptr);
            break;
        }
        case OAConfig::HBLOCK_TYPE::hbExtended:
        {
            ExtendedBlockHeader(ptr);
            break;
        }
        case OAConfig::HBLOCK_TYPE::hbExternal:
        {
            ExternalBlockHeader(ptr, label);
            break;
        }
        default: break;
    }
}

void ObjectAllocator::BasicBlockHeader(GenericObject* ptr) const
{
    unsigned char* header = GetHeaderAddress(ptr);

    unsigned int* alloc = reinterpret_cast<unsigned int*>(header);
    unsigned char* flag = reinterpret_cast<unsigned char*>(header + sizeof(unsigned int));

    *alloc = Stats_.Allocations_;
    *flag |= 0b1;   // setting the flag to 1
}

void ObjectAllocator::ExtendedBlockHeader(GenericObject* ptr) const
{
    unsigned char* header = GetHeaderAddress(ptr);

    size_t OFFSET = Config_.HBlockInfo_.additional_;
    unsigned short* useCount = reinterpret_cast<unsigned short*>(header + OFFSET);
    ++(*useCount);

    OFFSET += sizeof(short);
    unsigned int* alloc = reinterpret_cast<unsigned int*>(header + OFFSET);
    *alloc = Stats_.Allocations_;

    OFFSET += sizeof(unsigned int);
    unsigned char* flag = reinterpret_cast<unsigned char*>(header + OFFSET);
    *flag |= 0b1;
}

void ObjectAllocator::ExternalBlockHeader(GenericObject* ptr, char const* label) const
{
    unsigned char* header = GetHeaderAddress(ptr);

    MemBlockInfo* mbi = nullptr;

    try 
    {
        mbi = new MemBlockInfo{};
        size_t const LEN = strlen(label) + 1;
        mbi->label = new char[LEN];
        strcpy(mbi->label, label);
        *(mbi->label + LEN) = '\0';
        mbi->in_use = true;
        mbi->alloc_num = Stats_.Allocations_;
    }
    catch (std::bad_alloc const&)
    {
        throw OAException(OAException::E_NO_MEMORY, "No more memory!");
    }
    
    MemBlockInfo** mem = reinterpret_cast<MemBlockInfo**>(header);
    *mem = mbi;
}

void ObjectAllocator::ReleaseHeader(GenericObject* ptr) const
{
    unsigned char* header = GetHeaderAddress(ptr);
    char const* msg = "Multiple frees deducted";
    switch (Config_.HBlockInfo_.type_)
    {
        case OAConfig::HBLOCK_TYPE::hbNone:
        {
            if (Config_.DebugOn_)
            {
                unsigned char* c = reinterpret_cast<unsigned char*>(ptr) + sizeof(void*);
                if (FREED_PATTERN == *c)
                    throw OAException(OAException::E_MULTIPLE_FREE, msg);
            }
            break;
        }
        case OAConfig::HBLOCK_TYPE::hbBasic:
        {
            if (Config_.DebugOn_)
            {
                if( 0 == *( header + sizeof(unsigned int) ) )
                    throw OAException(OAException::E_MULTIPLE_FREE, msg);
            }
            memset(header, 0, OAConfig::BASIC_HEADER_SIZE);
            break;
        }
        case OAConfig::HBLOCK_TYPE::hbExtended:
        {
            if (Config_.DebugOn_)
            {
                if ( 0 == *( header + sizeof(unsigned int) + Config_.HBlockInfo_.additional_ + sizeof(unsigned short) ) )
                    throw OAException(OAException::E_MULTIPLE_FREE, msg);
            }
            memset(header + Config_.HBlockInfo_.additional_ + sizeof(unsigned short), 0, OAConfig::BASIC_HEADER_SIZE);
            break;
        }
        case OAConfig::HBLOCK_TYPE::hbExternal: 
        {
            MemBlockInfo** mem = reinterpret_cast<MemBlockInfo**>(header);
            if (Config_.DebugOn_)
            {
                if (!(*mem))
                    throw OAException(OAException::E_MULTIPLE_FREE, msg);
            }

            if (*mem)
            {
                if ((*mem)->label)
                {
                    delete[] (*mem)->label;
                    (*mem)->label = nullptr;
                }

                delete* mem;
                *mem = nullptr;
            }

            break;
        }
    }
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
    GenericObject* page{ nullptr };
    return WithinPage(ptr, page) && WithinDataSize(ptr, page) && AfterNextPointerAndLeftAlignment(ptr, page);
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
    // Any value other than PAD_PATTERN would means that the padding is corrupted
    for (size_t i = 0; i < Config_.PadBytes_; ++i)
        if (*(ptr + i) != PAD_PATTERN) return true;
    return false;
}

bool ObjectAllocator::WithinPage(unsigned char* ptr, GenericObject*& page) const
{
    page = PageList_;
    while (page)
    {
        unsigned char* pagePtr = reinterpret_cast<unsigned char*>(page);
        if (ptr >= pagePtr && ptr < pagePtr + Stats_.PageSize_) return true;
        page = page->Next;
    }
    return false;
}

bool ObjectAllocator::WithinDataSize(unsigned char* ptr, GenericObject* const& page) const
{
    unsigned char* head = reinterpret_cast<unsigned char*>(page) + sizeof(void*) + Config_.LeftAlignSize_;
    unsigned char* tail = reinterpret_cast<unsigned char*>(ptr) + Stats_.ObjectSize_ + Config_.PadBytes_ + Config_.InterAlignSize_;
    // If the address is divisible by the middle block size, then it's definitely a block within the page
    return !( (tail - head) % middleBlockSize );
}

bool ObjectAllocator::AfterNextPointerAndLeftAlignment(unsigned char* ptr, GenericObject* const& page) const
{
    unsigned char* head = reinterpret_cast<unsigned char*>(page);
    return static_cast<size_t>((ptr - head)) >= sizeof(void*) + static_cast<size_t>(Config_.LeftAlignSize_);
}
