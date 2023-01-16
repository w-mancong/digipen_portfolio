/*!*****************************************************************************
\file MemoryManager.h
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: Operating System
\par Assignment 4
\date 24-11-2022
\brief
This file contains function declaration for a memory manager clas that handles
contiguous memory
*******************************************************************************/
#include <iostream>
#include <list>
#include <iomanip>
#include <cstdlib>

/*!*****************************************************************************
    \brief Memory Manager class that handles allocation of memory for you
*******************************************************************************/
class MemoryManager
{
public:
    /*!*****************************************************************************
        \brief Constructor that takes in total number of bytes to be used for
        memory allocation.

        \param [in] total_bytes: Total Bytes that is avaliable for use using this object
    *******************************************************************************/
    MemoryManager(int total_bytes);

    /*!*****************************************************************************
        \brief Destructor
    *******************************************************************************/
    ~MemoryManager(void);

    /*!*****************************************************************************
        \brief Allocate a memory space from the pool of memory

        \param [in] bytes: Total number of bytes to allocate

        \return Pointer to the start of the element
    *******************************************************************************/
    void *allocate(int bytes);

    /*!*****************************************************************************
        \brief Deallocate memory allocated using memory manager

        \param [in] ptr: Memory to be deallocated

        \exception If memory cannot be found within the list, no memory will be deallocated
    *******************************************************************************/
    void deallocate(void *ptr);

    /*!*****************************************************************************
        \brief To dump the contents of the link list onto an output stream

        \param [out] out: source for ostream
    *******************************************************************************/
    void dump(std::ostream &out = std::cout);

private:
    /*!*****************************************************************************
        \brief Proxy struct to contain the neccessary data
    *******************************************************************************/
    struct Node
    {
        char* startAddress{ nullptr };
        int byteCount{ 0 };
        bool allocated{ false };
    };

    /*!*****************************************************************************
        \brief Print out the content of link list

        \param [in] startAddress: Address of the pointer to the head
        \param [in] byteCount: Total byte allocated for the memory pool
        \param [in] allocated: Is the node allocated?
        \param [out] out: source for ostream
    *******************************************************************************/
    void Print(void* startAddress, int byteCount, bool allocated, std::ostream& out);

    uint64_t m_TotalBytes{ 0 }, m_AllocatedBytes{ 0 };
    char* m_Memory{ nullptr };
    std::list<Node> m_NodeList{};
};