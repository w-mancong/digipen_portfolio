/*!*****************************************************************************
\file   ObjectAllocator.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: Data Structure
\par Section: A
\par Assignment 1
\date 26/01/2023
\brief
This file contains function declarations for ObjectAllocator class

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
//---------------------------------------------------------------------------
#ifndef OBJECTALLOCATORH
#define OBJECTALLOCATORH
//---------------------------------------------------------------------------

#include <string>

// If the client doesn't specify these:
static const int DEFAULT_OBJECTS_PER_PAGE = 4;  
static const int DEFAULT_MAX_PAGES = 3;

/*!
  Exception class
*/
class OAException
{
  public:
    /*!
      Possible exception codes
    */
    enum OA_EXCEPTION 
    {
      E_NO_MEMORY,      //!< out of physical memory (operator new fails)
      E_NO_PAGES,       //!< out of logical memory (max pages has been reached)
      E_BAD_BOUNDARY,   //!< block address is on a page, but not on any block-boundary
      E_MULTIPLE_FREE,  //!< block has already been freed
      E_CORRUPTED_BLOCK //!< block has been corrupted (pad bytes have been overwritten)
    };

    /*!
      Constructor

      \param ErrCode
        One of the 5 error codes listed above

      \param Message
        A message returned by the what method.
    */
    OAException(OA_EXCEPTION ErrCode, const std::string& Message) : error_code_(ErrCode), message_(Message) {};

    /*!
      Destructor
    */
    virtual ~OAException() {
    }

    /*!
      Retrieves the error code

      \return
        One of the 5 error codes.
    */
    OA_EXCEPTION code() const { 
      return error_code_; 
    }

    /*!
      Retrieves a human-readable string regarding the error.

      \return
        The NUL-terminated string representing the error.
    */
    virtual const char *what() const {
      return message_.c_str();
    }
  private:  
    OA_EXCEPTION error_code_; //!< The error code (one of the 5)
    std::string message_;     //!< The formatted string for the user.
};


/*!
  ObjectAllocator configuration parameters
*/
struct OAConfig
{
  static const size_t BASIC_HEADER_SIZE = sizeof(unsigned) + 1; //!< allocation number + flags
  static const size_t EXTERNAL_HEADER_SIZE = sizeof(void*);     //!< just a pointer

  /*!
    The different types of header blocks
  */
  enum HBLOCK_TYPE{hbNone, hbBasic, hbExtended, hbExternal};

  /*!
    POD that stores the information related to the header blocks.
  */
  struct HeaderBlockInfo
  {
    HBLOCK_TYPE type_;  //!< Which of the 4 header types to use?
    size_t size_;       //!< The size of this header
    size_t additional_; //!< How many user-defined additional bytes

    /*!
      Constructor

      \param type
        The kind of header blocks in use.

      \param additional
        The number of user-defined additional bytes required.

    */
    HeaderBlockInfo(HBLOCK_TYPE type = hbNone, unsigned additional = 0) : type_(type), size_(0), additional_(additional)
    {
      if (type_ == hbBasic)
        size_ = BASIC_HEADER_SIZE;
      else if (type_ == hbExtended) // alloc # + use counter + flag byte + user-defined
        size_ = sizeof(unsigned int) + sizeof(unsigned short) + sizeof(char) + additional_;
      else if (type_ == hbExternal)
        size_ = EXTERNAL_HEADER_SIZE;
    };
  };

  /*!
    Constructor

    \param UseCPPMemManager
      Determines whether or not to by-pass the OA.

    \param ObjectsPerPage
      Number of objects for each page of memory.

    \param MaxPages
      Maximum number of pages before throwing an exception. A value
      of 0 means unlimited.

    \param DebugOn
      Is debugging code on or off?

    \param PadBytes
      The number of bytes to the left and right of a block to pad with.

    \param HBInfo
      Information about the header blocks used

    \param Alignment
      The number of bytes to align on.
  */
  OAConfig(bool UseCPPMemManager = false,
           unsigned ObjectsPerPage = DEFAULT_OBJECTS_PER_PAGE, 
           unsigned MaxPages = DEFAULT_MAX_PAGES, 
           bool DebugOn = false, 
           unsigned PadBytes = 0,
           const HeaderBlockInfo &HBInfo = HeaderBlockInfo(),
           unsigned Alignment = 0) : UseCPPMemManager_(UseCPPMemManager),
                                     ObjectsPerPage_(ObjectsPerPage), 
                                     MaxPages_(MaxPages), 
                                     DebugOn_(DebugOn), 
                                     PadBytes_(PadBytes),
                                     HBlockInfo_(HBInfo),
                                     Alignment_(Alignment)
  {
    HBlockInfo_ = HBInfo;
    LeftAlignSize_ = 0;  
    InterAlignSize_ = 0;
  }

  bool UseCPPMemManager_;      //!< by-pass the functionality of the OA and use new/delete
  unsigned ObjectsPerPage_;    //!< number of objects on each page
  unsigned MaxPages_;          //!< maximum number of pages the OA can allocate (0=unlimited)
  bool DebugOn_;               //!< enable/disable debugging code (signatures, checks, etc.)
  unsigned PadBytes_;          //!< size of the left/right padding for each block
  HeaderBlockInfo HBlockInfo_; //!< size of the header for each block (0=no headers)
  unsigned Alignment_;         //!< address alignment of each block
  unsigned LeftAlignSize_;     //!< number of alignment bytes required to align first block
  unsigned InterAlignSize_;    //!< number of alignment bytes required between remaining blocks
};


/*!
  POD that holds the ObjectAllocator statistical info
*/
struct OAStats
{
  /*!
    Constructor
  */
  OAStats() : ObjectSize_(0), PageSize_(0), FreeObjects_(0), ObjectsInUse_(0), PagesInUse_(0),
                  MostObjects_(0), Allocations_(0), Deallocations_(0) {};

  size_t ObjectSize_;      //!< size of each object
  size_t PageSize_;        //!< size of a page including all headers, padding, etc.
  unsigned FreeObjects_;   //!< number of objects on the free list
  unsigned ObjectsInUse_;  //!< number of objects in use by client
  unsigned PagesInUse_;    //!< number of pages allocated
  unsigned MostObjects_;   //!< most objects in use by client at one time
  unsigned Allocations_;   //!< total requests to allocate memory
  unsigned Deallocations_; //!< total requests to free memory
};

/*!
  This allows us to easily treat raw objects as nodes in a linked list
*/
struct GenericObject
{
    GenericObject* Next{ nullptr }; //!< The next object in the list
};

/*!
  This is used with external headers
*/
struct MemBlockInfo
{
  bool in_use;        //!< Is the block free or in use?
  char *label;        //!< A dynamically allocated NUL-terminated string
  unsigned alloc_num; //!< The allocation number (count) of this block
};

/*!
  This class represents a custom memory manager
*/
class ObjectAllocator
{
  public:
    // Defined by the client (pointer to a block, size of block)
    typedef void (*DUMPCALLBACK)(const void *, size_t);     //!< Callback function when dumping memory leaks
    typedef void (*VALIDATECALLBACK)(const void *, size_t); //!< Callback function when validating blocks

    // Predefined values for memory signatures
    static const unsigned char UNALLOCATED_PATTERN = 0xAA; //!< New memory never given to the client
    static const unsigned char ALLOCATED_PATTERN   = 0xBB; //!< Memory owned by the client
    static const unsigned char FREED_PATTERN       = 0xCC; //!< Memory returned by the client
    static const unsigned char PAD_PATTERN         = 0xDD; //!< Pad signature to detect buffer over/under flow
    static const unsigned char ALIGN_PATTERN       = 0xEE; //!< For the alignment bytes

    /*!*****************************************************************************
        \brief ObjectAllocator's constructor

        \param [in] ObjecttSize: Size of the object
        \param [in] config: Configuration to construct this ObjectAllocator
    *******************************************************************************/
    ObjectAllocator(size_t ObjectSize, const OAConfig& config);

    /*!*****************************************************************************
        \brief Destructor for ObjectAllocator
    *******************************************************************************/
    ~ObjectAllocator();

    /*!*****************************************************************************
        \brief Allocate and return a free block of memory back to the Client.
               If UseCPPMemManager_ is true, the allocator will create a new pointer
               for the client using the new operator

        \param [in] label: If the Header is of External type, the allocator will
                create a pointer to store the string in label

        \return Pointer to the head of the first available address in the Page
             OR Pointer to the address allocated by the OS with the new operator
    *******************************************************************************/
    void *Allocate(const char *label = nullptr);

    /*!*****************************************************************************
        \brief Allow the address stored in Object be used as a free object for future
            OR release the memory allocated back to the OS (UseCPPMemManager_ => true)

        \param [in] Object: Pointer to an address in the page OR an address given
                by the OS when using the new operator
    *******************************************************************************/
    void Free(void *Object);

    /*!*****************************************************************************
        \brief Report the use of the memory if the block of memory in the page is
               allocated for the client based on the fn function

        \param [in] fn: Call back function specified by the user to report the
                memory usage

        \return The total number of memory block in use
    *******************************************************************************/
    unsigned DumpMemoryInUse(DUMPCALLBACK fn) const;

    /*!*****************************************************************************
        \brief Check and validate if any of the pages are corrupted. A page is corrupted
               when PadBytes allocated for padding are used

        \param [in] fn: Call back function specified by the user to validate the page

        \return The total number of blocks that are corrupted
    *******************************************************************************/
    unsigned ValidatePages(VALIDATECALLBACK fn) const;

    /*!*****************************************************************************
        \brief Free all pages that are not in use by the the client

        \return Total number of pages that have been freed
    *******************************************************************************/
    unsigned FreeEmptyPages();

    /*!*****************************************************************************
        \brief Setter function for OAConfig's DebugOn_

        \param [in] State: True=>Enable DebugOn_
                           False=>Disable DebugOn_
    *******************************************************************************/
    void SetDebugState(bool State);   // true=enable, false=disable

    /*!*****************************************************************************
        \brief Return the pointer to the first available object in the free list
    *******************************************************************************/
    const void *GetFreeList() const;  // returns a pointer to the internal free list

    /*!*****************************************************************************
        \brief Return the pointer to the head of the page list
    *******************************************************************************/
    const void *GetPageList() const;  // returns a pointer to the internal page list

    /*!*****************************************************************************
        \brief Return the configuration parameters
    *******************************************************************************/
    OAConfig GetConfig() const;       // returns the configuration parameters

    /*!*****************************************************************************
        \brief Return the statistics for the allocator
    *******************************************************************************/
    OAStats GetStats() const;         // returns the statistics for the allocator

    // Prevent copy construction and assignment
    ObjectAllocator(const ObjectAllocator &oa) = delete;            //!< Do not implement!
    ObjectAllocator &operator=(const ObjectAllocator &oa) = delete; //!< Do not implement!

  private:
    enum class Padding
    {
        Left,
        Right,
    };

    /*!*****************************************************************************
        \brief Helper function to increment the stats value when Allocate function
               is called
    *******************************************************************************/
    void IncrementStatsValue(void);

    /*!*****************************************************************************
        \brief Helper function to allocate a new page
    *******************************************************************************/
    void AllocateNewPage(void);

    /*!*****************************************************************************
        \brief Helper function that is called inside AllocateNewPage. This function
               assign the free objects into the FreeList_ for newly allocated pages
    *******************************************************************************/
    void AssignFreeListObjects(void);

    /*!*****************************************************************************
        \brief Helper function that is called inside AllocateNewPage. This function
               assign the byte signatures for the newly allocated pages.
    *******************************************************************************/
    void AssignByteSignatures(void);

    /*!*****************************************************************************
        \brief Helper function that is called inside AllocateNewPages. When new pages
               are allocated, this function set a default value for the header block
               based on the type
    *******************************************************************************/
    void DefaultHeaderBlockValue(void);

    /*!*****************************************************************************
        \brief Helper function to insert a free object into the the front of the FreeList_

        \param [in] obj: Pointer to address of the free object in the page
    *******************************************************************************/
    void AddObjectToFreeList(GenericObject* obj);

    /*!*****************************************************************************
        \brief Helper function to get the pointer to the head of where the Header
               block address is

        \param [in] ptr: Pointer to the head of the address to the object

        \return Pointer to the head of the Header block address
    *******************************************************************************/
    unsigned char* GetHeaderAddress(void* ptr) const;

    /*!*****************************************************************************
        \brief Helper function that is called inside Allocate. This function update
               the value for the different type of Header Block

        \param [in] ptr: Pointer to the address of the free object
        \param [in] label: Only used for HBLOCK_TYPE::hbExternal
    *******************************************************************************/
    void UpdateHeader(GenericObject* ptr, char const* label) const;

    /*!*****************************************************************************
        \brief Updates the value for pages that are Basic header block

        \param [in] ptr: Pointer to the address of the free object
    *******************************************************************************/
    void BasicBlockHeader(GenericObject* ptr) const;

    /*!*****************************************************************************
        \brief Updates the value for pages that are Extended header block

        \param [in] ptr: Pointer to the address of the free object
    *******************************************************************************/
    void ExtendedBlockHeader(GenericObject* ptr) const;

    /*!*****************************************************************************
        \brief Updates the value for pages that are External header block

        \param [in] ptr: Pointer to the address of the free object
        \param [in] label: Pointer to a string that will be stored inside MemBlockInfo

        \exception OAException::E_NO_MEMORY if new fails to allocate memory
    *******************************************************************************/
    void ExternalBlockHeader(GenericObject* ptr, char const * label) const;

    /*!*****************************************************************************
        \brief Helper function that will be called inside Free. This function will
               check for multiple frees if OAConfig::DebugOn_ is true. It also resets
               the value for Basic and Extended block, and releases the memory for
               External block.

        \param [in] ptr: Pointer to the address of the free object

        \exception OAException::E_MULTIPLE_FREE when OAConfig::DebugOn_ is true and
                   a free call was called on ptr
    *******************************************************************************/
    void ReleaseHeader(GenericObject* ptr) const;

    /*!*****************************************************************************
        \brief Helper function to set the byte signature for ptr

        \param [in] ptr: Pointer the the head of the address to assign the value of c to
        \param [in] c: Byte signature for the pointer
        \param [in] size: Total number of c to set
    *******************************************************************************/
    void UpdateByteSignature(unsigned char* ptr, unsigned char c, size_t size) const;

    /*!*****************************************************************************
        \brief Check if the address of ptr is within the boundary of the page

        \param [in] ptr: Pointer to the address to be checked with

        \return True: ptr is within the boundary of the page
                False: ptr is not within the boundary of the page
    *******************************************************************************/
    bool WithinMemoryBoundary(unsigned char* ptr) const;

    /*!*****************************************************************************
        \brief Get the pointer to the head of where the address of PadBytes_ is

        \param [in] ptr: Pointer to the address of the free object/address to be check with (for boundary checking)
        \param [in] p: Type of padding, left/right side

        \return Pointer to the head of where PadBytes_ is
    *******************************************************************************/
    unsigned char* GetPadding(GenericObject* ptr, Padding p) const;

    /*!*****************************************************************************
        \brief Check if the padding is corrupted. Padding is corrupted when the PadBytes_
               do not have the byte signature of PAD_PATTERN

        \param [in] ptr: Pointer to the address of where the paddings are

        \return True if padding is corrupted, else false
    *******************************************************************************/
    bool IsPaddingCorrupted(unsigned char* ptr) const;

    /*!*****************************************************************************
        \brief Check if ptr is an address within PageList_

        \param [in] ptr: Pointer to be check if is an address within PageList_
        \param [in, out] page: Store the address of the page if ptr is found to be within it

        \return True if ptr is within any of the PageList_, else false
    *******************************************************************************/
    bool WithinPage(unsigned char* ptr, GenericObject*& page) const;

    /*!*****************************************************************************
        \brief Check if ptr stores the exact size of the Middle Block

        \param [in] ptr: Pointer to be check with
        \param [in] page: Pointer that stores the address of the page that contains ptr

        \return True if the data block is divisible by middleBlockSize, else false
    *******************************************************************************/
    bool WithinDataSize(unsigned char* ptr, GenericObject* const& page) const;

    /*!*****************************************************************************
        \brief Check to make sure that ptr is an address after NextPointer and left
               alignment bytes

        \param [in] ptr: Pointer to be check with
        \param [in] page: Pointer that stores the address of the page that contain ptr

        \return True if ptr is after NextPointer and left alignment bytes, else false
    *******************************************************************************/
    bool AfterNextPointerAndLeftAlignment(unsigned char* ptr, GenericObject* const& page) const;

    /*!*****************************************************************************
        \brief Check if an object is in use

        \param [in] ptr: Pointer to the address of the free object

        \return True if ptr is in use, else false
    *******************************************************************************/
    bool IsObjectInUse(GenericObject* ptr) const;

    /*!*****************************************************************************
        \brief Helper function that is called inside FreeEmptyPages. Remove all free
               objects inside page from the FreeList_

        \param [in] page: Pointer to the page that contains all the objects
               that are not in used, and to be removed from the FreeList_
    *******************************************************************************/
    void RemoveFromFreeList(GenericObject* page);

    // Some "suggested" members (only a suggestion!)
    GenericObject *PageList_{ nullptr }; //!< the beginning of the list of pages
    GenericObject *FreeList_{ nullptr }; //!< the beginning of the list of objects
    
    // Lots of other private stuff...
    OAConfig Config_{};
    OAStats  Stats_{};

    size_t middleBlockSize{}, leftAlignSize{}, dataBlockSize{};
};

#endif
