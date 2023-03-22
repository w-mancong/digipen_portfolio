/*!*****************************************************************************
\file   ObjectAllocator.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: Data Structure
\par Section: A
\par Assignment 1
\date 26/1/2023
\brief
This file contains function definition for ObjectAllocator

A standard page of 3 objects for this assignment will look like this:
Page: NP|AL|H|P|S|P|AI|H|P|S|P|AI|H|P|S|P|
            ^          ^          ^

NP: Pointer to the next Page
AL: Left alignment bytes
H:  Header block => Basic, Extended, External
P:  PadBytes => Used to check for any corruption in the page
S:  Size of object => if the object is an int, this value will be 4 bytes (64 bit computer)
AI: Inter alignment bytes
*******************************************************************************/
#include <cstring>
#include "ObjectAllocator.h"

namespace
{
    /*!*****************************************************************************
        \brief Calculate the maximum alignment byte required

        \param [in] n: Size of the block
        \param [in] align: The alignment for this block

        \return The maximum alignment byte required 
    *******************************************************************************/
    size_t AlignByte(size_t n, size_t align)
    {
        if (!align) return n;
        size_t r = n % align != 0 ? 1ULL : 0ULL;  // Getting the remainer
        return align * ((n / align) + r);
    }
}

/*!*****************************************************************************
    \brief ObjectAllocator's constructor

    \param [in] ObjecttSize: Size of the object
    \param [in] config: Configuration to construct this ObjectAllocator
*******************************************************************************/
ObjectAllocator::ObjectAllocator(size_t ObjectSize, const OAConfig &config) : Config_(config)
{
    Stats_.ObjectSize_ = ObjectSize;
    /*
        Page size consist of:
        Pointer to the next block, left alignment, middle block
        Middle block -> Header block, Padding, Size of Object, Padding, Inter alignment
    */
    leftAlignSize = sizeof(void*) + Config_.HBlockInfo_.size_ + Config_.PadBytes_;
    dataBlockSize = Stats_.ObjectSize_ + (Config_.PadBytes_ * 2ULL) + Config_.HBlockInfo_.size_;

    size_t const LEFT_ALIGNMENT_OFFSET  = AlignByte(leftAlignSize,  static_cast<size_t>(Config_.Alignment_)),
                 INTER_ALIGNMENT_OFFSET = AlignByte(dataBlockSize, static_cast<size_t>(Config_.Alignment_));

    Config_.LeftAlignSize_  = static_cast<unsigned int>(LEFT_ALIGNMENT_OFFSET  - leftAlignSize);
    Config_.InterAlignSize_ = static_cast<unsigned int>(INTER_ALIGNMENT_OFFSET - dataBlockSize);
    middleBlockSize = Config_.HBlockInfo_.size_ + (Config_.PadBytes_ * 2ULL) + Stats_.ObjectSize_ + Config_.InterAlignSize_;
    Stats_.PageSize_ = sizeof(void*) + Config_.LeftAlignSize_ + (middleBlockSize * Config_.ObjectsPerPage_) - Config_.InterAlignSize_;
   
    AllocateNewPage();
}

/*!*****************************************************************************
    \brief Destructor for ObjectAllocator
*******************************************************************************/
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

        delete[] reinterpret_cast<unsigned char*>(ptr);
        ptr = n;
    }
}

/*!*****************************************************************************
    \brief Allocate and return a free block of memory back to the Client. 
           If UseCPPMemManager_ is true, the allocator will create a new pointer
           for the client using the new operator

    \param [in] label: If the Header is of External type, the allocator will
            create a pointer to store the string in label

    \return Pointer to the head of the first available address in the Page
         OR Pointer to the address allocated by the OS with the new operator
*******************************************************************************/
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

/*!*****************************************************************************
    \brief Allow the address stored in Object be used as a free object for future
        OR release the memory allocated back to the OS (UseCPPMemManager_ => true)

    \param [in] Object: Pointer to an address in the page OR an address given
            by the OS when using the new operator
*******************************************************************************/
void ObjectAllocator::Free(void *Object)
{
    ++Stats_.Deallocations_;

    if (Config_.UseCPPMemManager_)
    {
        delete[] reinterpret_cast<unsigned char*>(Object);
        return;
    }

    GenericObject* ptr = reinterpret_cast<GenericObject*>(Object);
    //ptr->Next = nullptr;

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
    UpdateByteSignature(reinterpret_cast<unsigned char*>(ptr) + sizeof(void*), FREED_PATTERN, Stats_.ObjectSize_ - sizeof(void*));
}

/*!*****************************************************************************
    \brief Report the use of the memory if the block of memory in the page is
           allocated for the client based on the fn function

    \param [in] fn: Call back function specified by the user to report the
            memory usage

    \return The total number of memory block in use
*******************************************************************************/
unsigned ObjectAllocator::DumpMemoryInUse(DUMPCALLBACK fn) const
{
    // Empty Page
    if (!PageList_)
        return 0;

    unsigned int numBlockMemUsed{};

    GenericObject* page = PageList_;
    while (page)
    {
        unsigned char* header = reinterpret_cast<unsigned char*>(page);

        for (size_t i = 0; i < Config_.ObjectsPerPage_; ++i)
        {
            size_t const OFFSET = sizeof(void*) + Config_.LeftAlignSize_ + (middleBlockSize * i);
            GenericObject* ptr = reinterpret_cast<GenericObject*>(header + OFFSET);

            if (IsObjectInUse(ptr))
            {
                ++numBlockMemUsed;
                ptr = reinterpret_cast<GenericObject*>(header + OFFSET + Config_.HBlockInfo_.size_ + Config_.PadBytes_);
                fn(ptr, Stats_.ObjectSize_);
            }
        }

        page = page->Next;
    }

    return numBlockMemUsed;
}

/*!*****************************************************************************
    \brief Check and validate if any of the pages are corrupted. A page is corrupted
           when PadBytes allocated for padding are used

    \param [in] fn: Call back function specified by the user to validate the page

    \return The total number of blocks that are corrupted
*******************************************************************************/
unsigned ObjectAllocator::ValidatePages(VALIDATECALLBACK fn) const
{
    if (!Config_.DebugOn_ || Config_.PadBytes_ == 0)
        return 0;
    unsigned int numBlockCorrupted{};

    GenericObject* page = PageList_;
    while (page)
    {
        unsigned char* header = reinterpret_cast<unsigned char*>(page);

        for (size_t i = 0; i < Config_.ObjectsPerPage_; ++i)
        {
            size_t const OFFSET = sizeof(void*) + Config_.LeftAlignSize_ + Config_.HBlockInfo_.size_ + Config_.PadBytes_ + (middleBlockSize * i);
            GenericObject* const ptr = reinterpret_cast<GenericObject*>(header + OFFSET);

            // If either left/right padding is corrupted, then run the callback function
            if ( IsPaddingCorrupted( GetPadding(ptr, Padding::Left) ) || IsPaddingCorrupted( GetPadding(ptr, Padding::Right) ) )
            {
                ++numBlockCorrupted;
                fn(ptr, Stats_.ObjectSize_);
            }
        }

        page = page->Next;
    }

    return numBlockCorrupted;
}

/*!*****************************************************************************
    \brief Free all pages that are not in use by the the client

    \return Total number of pages that have been freed 
*******************************************************************************/
unsigned ObjectAllocator::FreeEmptyPages()
{
    // If page list is empty, return 0
    if (!PageList_)
        return 0;

    unsigned int numPagesFreed{};

    /*
        1) Loop through each page and see if the object is in the free list
        -> if all $(ObjectsPerPage_) are in the FreeList_ means this page should be freed
        2) Remove the objects in the page from the FreeList_
        3) Remove the empty page from PageList_
        4) Return the number of page freed
    */

    GenericObject* page = PageList_, *prevPage{ nullptr };

    while (page)
    {
        unsigned char* header = reinterpret_cast<unsigned char*>(page);

        size_t unusedObjects{};

        for (size_t i = 0; i < Config_.ObjectsPerPage_; ++i)
        {
            size_t const OFFSET = sizeof(void*) + Config_.LeftAlignSize_ + (middleBlockSize * i);
            GenericObject* const ptr = reinterpret_cast<GenericObject*>(header + OFFSET);

            if (!IsObjectInUse(ptr))
                ++unusedObjects;
        }

        // the number of unused objects is the same as the number of objects per page
        if (unusedObjects == Config_.ObjectsPerPage_)
        {   // This is the page that is to be freed
            ++numPagesFreed;

            /*
                Then remove this page from PageList_
                1) if prevPage is nullptr, means im removing the head
                2) else need to relink prevPage's Next ptr to page's Next ptr
            */ 
            if (prevPage)   // removing a page that is between two other
                prevPage->Next = page->Next;
            else    // removing a page which is the head
                PageList_ = PageList_->Next;

            // Remove all the objects in this page from the free list
            RemoveFromFreeList(page);

            GenericObject* tmp = page->Next;

            // After re-arranging the PageList_, now delete this page
            delete[] reinterpret_cast<unsigned char*>(page);
            page = tmp;
        }

        if (unusedObjects < Config_.ObjectsPerPage_)
        {
            prevPage = page;
            page = page->Next;
        }
    }

    return numPagesFreed;
}

/*!*****************************************************************************
    \brief Setter function for OAConfig's DebugOn_

    \param [in] State: True=>Enable DebugOn_
                       False=>Disable DebugOn_
*******************************************************************************/
void ObjectAllocator::SetDebugState(bool State) // true=enable, false=disable
{
    Config_.DebugOn_ = State;
}

/*!*****************************************************************************
    \brief Return the pointer to the first available object in the free list
*******************************************************************************/
const void *ObjectAllocator::GetFreeList() const // returns a pointer to the internal free list
{
    return FreeList_;
}

/*!*****************************************************************************
    \brief Return the pointer to the head of the page list
*******************************************************************************/
const void *ObjectAllocator::GetPageList() const // returns a pointer to the internal page list
{
    return PageList_;
}

/*!*****************************************************************************
    \brief Return the configuration parameters
*******************************************************************************/
OAConfig ObjectAllocator::GetConfig() const      // returns the configuration parameters
{
    return Config_;
}

/*!*****************************************************************************
    \brief Return the statistics for the allocator
*******************************************************************************/
OAStats ObjectAllocator::GetStats() const        // returns the statistics for the allocator
{
    return Stats_;
}

/*!*****************************************************************************
    \brief Helper function to increment the stats value when Allocate function
           is called
*******************************************************************************/
void ObjectAllocator::IncrementStatsValue(void)
{
    --Stats_.FreeObjects_, ++Stats_.Allocations_, ++Stats_.ObjectsInUse_;
    if (Stats_.MostObjects_ < Stats_.ObjectsInUse_)
        Stats_.MostObjects_ = Stats_.ObjectsInUse_;
}

/*!*****************************************************************************
    \brief Helper function to allocate a new page
*******************************************************************************/
void ObjectAllocator::AllocateNewPage(void)
{
    if (Stats_.PagesInUse_ >= Config_.MaxPages_)
        throw OAException(OAException::E_NO_PAGES, "Out of Pages!");

    // Allocate a new page and insert it at the front of the PageList_
    unsigned char *ptr = nullptr;
    try
    {
        ptr = new unsigned char[Stats_.PageSize_]{};
            
        // Allocating spaces for page
        if (PageList_)
        {   
            GenericObject* head = reinterpret_cast<GenericObject*>(ptr);
            head->Next = PageList_;
            PageList_ = head;
        }
        else
        {   // PageList_ is nullptr, so this very first page will be the head
            PageList_ = reinterpret_cast<GenericObject*>(ptr);
            PageList_->Next = nullptr;
        }
    }
    catch (std::bad_alloc const&)
    {
        throw OAException(OAException::E_NO_MEMORY, "No more memory!");
    }

    ++Stats_.PagesInUse_;

    AssignFreeListObjects();
    AssignByteSignatures();
    DefaultHeaderBlockValue();
}

/*!*****************************************************************************
    \brief Helper function that is called inside AllocateNewPage. This function
           assign the free objects into the FreeList_ for newly allocated pages
*******************************************************************************/
void ObjectAllocator::AssignFreeListObjects(void)
{
    unsigned char* head = reinterpret_cast<unsigned char*>( PageList_ );
    size_t const OFFSET = sizeof(void*) + static_cast<size_t>(Config_.LeftAlignSize_) + static_cast<size_t>(Config_.HBlockInfo_.size_) + static_cast<size_t>(Config_.PadBytes_);

    for (size_t i = 0; i < Config_.ObjectsPerPage_; ++i)
    {
        GenericObject* ptr = reinterpret_cast<GenericObject*>(head + OFFSET + (middleBlockSize * i));
        AddObjectToFreeList(ptr);
    }
}

/*!*****************************************************************************
    \brief Helper function that is called inside AllocateNewPage. This function 
           assign the byte signatures for the newly allocated pages.
*******************************************************************************/
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

/*!*****************************************************************************
    \brief Helper function that is called inside AllocateNewPages. When new pages
           are allocated, this function set a default value for the header block
           based on the type
*******************************************************************************/
void ObjectAllocator::DefaultHeaderBlockValue(void)
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
        default: break;
    }
}

/*!*****************************************************************************
    \brief Helper function to insert a free object into the the front of the FreeList_

    \param [in] obj: Pointer to address of the free object in the page
*******************************************************************************/
void ObjectAllocator::AddObjectToFreeList(GenericObject* obj)
{
    ++Stats_.FreeObjects_;

    if (!FreeList_)
    {
        FreeList_ = obj;
        FreeList_->Next = nullptr;
        return;
    }

    GenericObject* ptr = FreeList_;
    FreeList_ = obj;
    obj->Next = ptr;
}

/*!*****************************************************************************
    \brief Helper function to get the pointer to the head of where the Header
           block address is
    
    \param [in] ptr: Pointer to the head of the address to the object

    \return Pointer to the head of the Header block address
*******************************************************************************/
unsigned char* ObjectAllocator::GetHeaderAddress(void* ptr) const
{
    return reinterpret_cast<unsigned char*>(ptr) - Config_.PadBytes_ - Config_.HBlockInfo_.size_;
}

/*!*****************************************************************************
    \brief Helper function that is called inside Allocate. This function update
           the value for the different type of Header Block

    \param [in] ptr: Pointer to the address of the free object
    \param [in] label: Only used for HBLOCK_TYPE::hbExternal
*******************************************************************************/
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

/*!*****************************************************************************
    \brief Updates the value for pages that are Basic header block

    \param [in] ptr: Pointer to the address of the free object
*******************************************************************************/
void ObjectAllocator::BasicBlockHeader(GenericObject* ptr) const
{
    unsigned char* header = GetHeaderAddress(ptr);

    unsigned int* alloc = reinterpret_cast<unsigned int*>(header);
    unsigned char* flag = reinterpret_cast<unsigned char*>(header + sizeof(unsigned int));

    *alloc = Stats_.Allocations_;
    *flag |= 0b1;   // setting the flag to 1
}

/*!*****************************************************************************
    \brief Updates the value for pages that are Extended header block

    \param [in] ptr: Pointer to the address of the free object
*******************************************************************************/
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

/*!*****************************************************************************
    \brief Updates the value for pages that are External header block

    \param [in] ptr: Pointer to the address of the free object
    \param [in] label: Pointer to a string that will be stored inside MemBlockInfo

    \exception OAException::E_NO_MEMORY if new fails to allocate memory
*******************************************************************************/
void ObjectAllocator::ExternalBlockHeader(GenericObject* ptr, char const* label) const
{
    unsigned char* header = GetHeaderAddress(ptr);

    MemBlockInfo* mbi = nullptr;

    try 
    {
        mbi = new MemBlockInfo{};

        if (label)
        {
            size_t const LEN = strlen(label) + 1;
            mbi->label = new char[LEN];
            strcpy(mbi->label, label);
            *(mbi->label + LEN) = '\0';
        }

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

/*!*****************************************************************************
    \brief Helper function that will be called inside Free. This function will
           check for multiple frees if OAConfig::DebugOn_ is true. It also resets
           the value for Basic and Extended block, and releases the memory for
           External block.

    \param [in] ptr: Pointer to the address of the free object

    \exception OAException::E_MULTIPLE_FREE when OAConfig::DebugOn_ is true and
               a free call was called on ptr 
*******************************************************************************/
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

/*!*****************************************************************************
    \brief Helper function to set the byte signature for ptr

    \param [in] ptr: Pointer the the head of the address to assign the value of c to
    \param [in] c: Byte signature for the pointer
    \param [in] size: Total number of c to set
*******************************************************************************/
void ObjectAllocator::UpdateByteSignature(unsigned char* ptr, unsigned char c, size_t size) const
{
    if(Config_.DebugOn_)
        memset(ptr, c, size);
}

/*!*****************************************************************************
    \brief Check if the address of ptr is within the boundary of the page

    \param [in] ptr: Pointer to the address to be checked with

    \return True: ptr is within the boundary of the page
            False: ptr is not within the boundary of the page
*******************************************************************************/
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

/*!*****************************************************************************
    \brief Get the pointer to the head of where the address of PadBytes_ is

    \param [in] ptr: Pointer to the address of the free object/address to be check with (for boundary checking)
    \param [in] p: Type of padding, left/right side

    \return Pointer to the head of where PadBytes_ is
*******************************************************************************/
unsigned char* ObjectAllocator::GetPadding(GenericObject* ptr, Padding p) const
{
    switch (p)
    {
        case Padding::Left : return reinterpret_cast<unsigned char*>(ptr) - Config_.PadBytes_;
        case Padding::Right: return reinterpret_cast<unsigned char*>(ptr) + Stats_.ObjectSize_;
        default: return nullptr;
    }
}

/*!*****************************************************************************
    \brief Check if the padding is corrupted. Padding is corrupted when the PadBytes_
           do not have the byte signature of PAD_PATTERN

    \param [in] ptr: Pointer to the address of where the paddings are

    \return True if padding is corrupted, else false
*******************************************************************************/
bool ObjectAllocator::IsPaddingCorrupted(unsigned char* ptr) const
{
    // Any value other than PAD_PATTERN would means that the padding is corrupted
    for (size_t i = 0; i < Config_.PadBytes_; ++i)
    {
        if (*(ptr + i) != PAD_PATTERN) 
            return true;
    }
    return false;
}

/*!*****************************************************************************
    \brief Check if ptr is an address within PageList_

    \param [in] ptr: Pointer to be check if is an address within PageList_
    \param [in, out] page: Store the address of the page if ptr is found to be within it

    \return True if ptr is within any of the PageList_, else false 
*******************************************************************************/
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

/*!*****************************************************************************
    \brief Check if ptr stores the exact size of the Middle Block

    \param [in] ptr: Pointer to be check with
    \param [in] page: Pointer that stores the address of the page that contains ptr

    \return True if the data block is divisible by middleBlockSize, else false
*******************************************************************************/
bool ObjectAllocator::WithinDataSize(unsigned char* ptr, GenericObject* const& page) const
{
    unsigned char* head = reinterpret_cast<unsigned char*>(page) + sizeof(void*) + Config_.LeftAlignSize_;
    unsigned char* tail = reinterpret_cast<unsigned char*>(ptr) + Stats_.ObjectSize_ + Config_.PadBytes_ + Config_.InterAlignSize_;
    // If the address is divisible by the middle block size, then it's definitely a block within the page
    return !( (tail - head) % middleBlockSize );
}

/*!*****************************************************************************
    \brief Check to make sure that ptr is an address after NextPointer and left
           alignment bytes

    \param [in] ptr: Pointer to be check with
    \param [in] page: Pointer that stores the address of the page that contain ptr

    \return True if ptr is after NextPointer and left alignment bytes, else false
*******************************************************************************/
bool ObjectAllocator::AfterNextPointerAndLeftAlignment(unsigned char* ptr, GenericObject* const& page) const
{
    unsigned char* head = reinterpret_cast<unsigned char*>(page);
    return static_cast<size_t>((ptr - head)) >= sizeof(void*) + static_cast<size_t>(Config_.LeftAlignSize_);
}

/*!*****************************************************************************
    \brief Check if an object is in use

    \param [in] ptr: Pointer to the address of the free object

    \return True if ptr is in use, else false
*******************************************************************************/
bool ObjectAllocator::IsObjectInUse(GenericObject* ptr) const
{
    switch (Config_.HBlockInfo_.type_)
    {
        case OAConfig::HBLOCK_TYPE::hbNone:
        {   // Search the entire FreeList_ to make sure the address of ptr is not part of it
            GenericObject* obj = FreeList_;
            unsigned char* header = reinterpret_cast<unsigned char*>(ptr) + Config_.PadBytes_;

            // If ptr is found inside the FreeList_, means that the object is not in use
            while (obj)
            {
                // checking the address to see if it matches
                if (reinterpret_cast<void*>(obj) == reinterpret_cast<void*>(header))
                    return false;
                obj = obj->Next;
            }

            return true;
        }
        case OAConfig::HBLOCK_TYPE::hbBasic:
        case OAConfig::HBLOCK_TYPE::hbExtended:
        {   // For both basic and extended, return the value of the flag bit
            unsigned char* flag = reinterpret_cast<unsigned char*>(ptr) + Config_.HBlockInfo_.size_ - 1ULL;
            return *flag;
        }
        case OAConfig::HBLOCK_TYPE::hbExternal:
        {
            MemBlockInfo** mem = reinterpret_cast<MemBlockInfo**>(ptr);
            // if mem is not nullptr means this block of memory is in use
            return static_cast<bool>(*mem);
        }
    }
    return false;
}

/*!*****************************************************************************
    \brief Helper function that is called inside FreeEmptyPages. Remove all free
           objects inside page from the FreeList_

    \param [in] page: Pointer to the page that contains all the objects
           that are not in used, and to be removed from the FreeList_
*******************************************************************************/
void ObjectAllocator::RemoveFromFreeList(GenericObject* page)
{
    unsigned char* header = reinterpret_cast<unsigned char*>(page);

    for (size_t i = 0; i < Config_.ObjectsPerPage_; ++i)
    {
        size_t const OFFSET = sizeof(void*) + Config_.LeftAlignSize_ + Config_.HBlockInfo_.size_ + Config_.PadBytes_ + (middleBlockSize * i);
        GenericObject* ptr = reinterpret_cast<GenericObject*>(header + OFFSET);

        GenericObject* list = FreeList_, *prev{ nullptr };
        while (list)
        {
            // The current list object is the one we're removing from the list
            if (ptr == list)
            {
                if (prev)
                    prev->Next = list->Next;
                else
                    FreeList_ = FreeList_->Next;

                --Stats_.FreeObjects_;

                break;
            }

            prev = list;
            list = list->Next;
        }
    }
    --Stats_.PagesInUse_;
}
