/*!*****************************************************************************
\file sllist.cpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 4
\date 04-02-2022
\brief
This program provide and ADT interface for user to create a link list to store
int values.
*******************************************************************************/
#ifndef SLLIST_HPP_
#define SLLIST_HPP_    

#include <cstddef>

namespace hlp2
{
    struct node;
    struct sllist;

    /*!*****************************************************************************
	\brief
        Return integral data stored inside node pointer p
    \param[in] p
        Pointer to a node to retrieve it's integral data
    \return 
        Integral data node pointer p is storing
    *******************************************************************************/
    int         data(node const *p);        

    /*!*****************************************************************************
	\brief
        Set node pointer p's integral data
    \param[in, out] p
        Pointer to a node to store the integral data
    \param[in] newVal
        Integral data to be stored
    *******************************************************************************/
    void        data(node *p, int newVal);

    /*!*****************************************************************************
	\brief
        Helper function that returns the pointer to the next node
    \param[in] p
        Node pointer storing the next pointer node
    \return
        Pointer to the next node
    *******************************************************************************/
    node*       next(node *p);          

    /*!*****************************************************************************
	\brief
        Helper function that returns the pointer to the next node
    \param[in] p
        Node pointer storing the next pointer node        
    \return
        Pointer to the next node
    *******************************************************************************/
    node const* next(node const *p);        

    /*!*****************************************************************************
	\brief
        Allocate memory for sslist on the heap
    \return
        Pointer to sllist        
    *******************************************************************************/
    sllist*     construct(void);

    /*!*****************************************************************************
	\brief
        Deallocate all memory from the head to the last element in the linked-list
    \param[in] ptr_sll
        Pointer to sllist to have it's memory deallocated        
    *******************************************************************************/
    void        destruct(sllist *ptr_sll);

    /*!*****************************************************************************
	\brief
        Check if the linked-list is empty
    \param[in] ptr_sll
        Pointer to sllist to check if the linked-list is empty
    \return
        Returns true if the linked-list is empty, false if it's not
    *******************************************************************************/
    bool        empty(sllist const *ptr_sll);

    /*!*****************************************************************************
	\brief
        Count the total elements inside of the linked-list
    \param[in] ptr_sll
        Pointer that stores the head node pointer
    \return 
        Total elements in the linked-list
    *******************************************************************************/
    size_t      size(sllist const *ptr_sll);

    /*!*****************************************************************************
	\brief
        Add a new node to the front of the linked-list
    \param[in] ptr_sll
        Pointer that stores the head node pointer
    \param[in] value
        Integral value to be stored
    *******************************************************************************/
    void        push_front(sllist *ptr_sll, int value);

    /*!*****************************************************************************
	\brief
        Add a new node to the back of the linked-list
    \param[in] ptr_sll
        Pointer that stores the head node pointer
    \param[in] value
        Integral data to be stored
    *******************************************************************************/
    void        push_back(sllist *ptr_sll, int value);

    /*!*****************************************************************************
	\brief
        Remove the first instance of node pointer that have the same integral data
        as value
    \param[in] ptr_sll
        Pointer that stores the head node pointer
    \param[in] value
        Integral value to be removed
    *******************************************************************************/
    void        remove_first(sllist *ptr_sll, int value);

    /*!*****************************************************************************
	\brief
        Insert a new node pointer based on the position inside the linked-list
    \param[in]  ptr_sll
        Pointer that stores the head node pointers
    \param[in] value
        Integral data to be stored
    \param[in] index
        Position to insert a new node pointer
    *******************************************************************************/
    void        insert(sllist *ptr_sll, int value, size_t index);

    /*!*****************************************************************************
	\brief
        Returns the first node in the linked-list
    \param[in] ptr_sll
        Pointer that stores the head node pointer
    \return
        Pointer to the first node        
    *******************************************************************************/
    node*       front(sllist *ptr_sll);

    /*!*****************************************************************************
	\brief
        Return the first node in the linked-list for reference only
    \param[in] ptr_sll
        Pointer that stores the head node pointer
    \return
        Pointer to the first node
    *******************************************************************************/
    node const* front(sllist const *ptr_sll);

    /*!*****************************************************************************
	\brief
        Find the first node pointer that have the same integral value
    \param[in] ptr_sll
        Pointer that stores the head node pointer
    \param[in] value
        Integral data to be searched for inside the linked-list
    \return
        Pointer to the first node that stores the same integral data as value
    *******************************************************************************/
    node*       find(sllist const *ptr_sll, int value);
}

#endif