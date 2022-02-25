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
#include "sllist.hpp"

namespace
{
    /*!*****************************************************************************
	\brief
        Creates a new node that exist in the free store
	\param[in] value
        Integral value to be stored
    \param[in] next
        Pointer to the next node
    \return
        A node that stores value with a pointer pointing to next
    *******************************************************************************/
    hlp2::node* create_node(int value, hlp2::node* next = nullptr);
}

namespace Helper
{
    /*!*****************************************************************************
	\brief
        Does a recursive function to add a new node at the back
    \param[in] curr
        Pointer to the current node to be checked
	\param[in] value
        Integral value to be stored
    *******************************************************************************/
    void push_back(hlp2::node*& curr, int value);

    /*!*****************************************************************************
	\brief
        Insert a new node at position specified by index
    \param[in] curr
        Pointer to the next node
	\param[in] value
        Pointer to the current node to be checked
    \param[in] index
        Current position in the linked-list
    *******************************************************************************/
    void insert(hlp2::node*& curr, int value, int index);
}

namespace hlp2
{
    struct node
    {
        int value;      // data portion         
        node* next;     // pointer portion
    };

    struct sllist
    {
        node* head;
    };

    /*!*****************************************************************************
	\brief
        Return integral data stored inside node pointer p
    \param[in] p
        Pointer to a node to retrieve it's integral data
    \return 
        Integral data node pointer p is storing
    *******************************************************************************/
    int         data(node const *p)         { return p->value; }
    
    /*!*****************************************************************************
	\brief
        Set node pointer p's integral data
    \param[in, out] p
        Pointer to a node to store the integral data
    \param[in] newVal
        Integral data to be stored
    *******************************************************************************/
    void        data(node *p, int newVal)   { p->value = newVal; }

    /*!*****************************************************************************
	\brief
        Helper function that returns the pointer to the next node
    \param[in] p
        Node pointer storing the next pointer node
    \return
        Pointer to the next node
    *******************************************************************************/
    node*       next(node *p)               { return p->next; }

    /*!*****************************************************************************
	\brief
        Helper function that returns the pointer to the next node
    \param[in] p
        Node pointer storing the next pointer node        
    \return
        Pointer to the next node
    *******************************************************************************/
    node const* next(node const *p)         { return p->next; }

    /*!*****************************************************************************
	\brief
        Allocate memory for sslist on the heap
    \return
        Pointer to sllist        
    *******************************************************************************/
    sllist* construct(void)
    {
        return new sllist { nullptr };
    }

    /*!*****************************************************************************
	\brief
        Deallocate all memory from the head to the last element in the linked-list
    \param[in] ptr_sll
        Pointer to sllist to have it's memory deallocated        
    *******************************************************************************/
    void destruct(sllist *ptr_sll)
    {
        for(auto next_ptr = ptr_sll->head; next_ptr;)
        {
            auto temp = next(next_ptr);
            delete next_ptr;
            next_ptr = nullptr;
            next_ptr = temp;
        }
        delete ptr_sll;
    }
    
    /*!*****************************************************************************
	\brief
        Check if the linked-list is empty
    \param[in] ptr_sll
        Pointer to sllist to check if the linked-list is empty
    \return
        Returns true if the linked-list is empty, false if it's not
    *******************************************************************************/
    bool empty(sllist const *ptr_sll)
    {
        return !ptr_sll;
    }

    /*!*****************************************************************************
	\brief
        Count the total elements inside of the linked-list
    \param[in] ptr_sll
        Pointer that stores the head node pointer
    \return 
        Total elements in the linked-list
    *******************************************************************************/
    size_t size(sllist const *ptr_sll)
    {
        size_t count{};
        for(auto next_ptr = ptr_sll->head; next_ptr; next_ptr = next(next_ptr))
            ++count;
        return count;
    }

    /*!*****************************************************************************
	\brief
        Add a new node to the front of the linked-list
    \param[in] ptr_sll
        Pointer that stores the head node pointer
    \param[in] value
        Integral value to be stored
    *******************************************************************************/
    void push_front(sllist *ptr_sll, int value)
    {
        ptr_sll->head = create_node(value, ptr_sll->head);
    }

    /*!*****************************************************************************
	\brief
        Add a new node to the back of the linked-list
    \param[in] ptr_sll
        Pointer that stores the head node pointer
    \param[in] value
        Integral data to be stored
    *******************************************************************************/
    void push_back(sllist *ptr_sll, int value)
    {
        Helper::push_back(ptr_sll->head, value);
    }

    /*!*****************************************************************************
	\brief
        Remove the first instance of node pointer that have the same integral data
        as value
    \param[in] ptr_sll
        Pointer that stores the head node pointer
    \param[in] value
        Integral value to be removed
    *******************************************************************************/
    void remove_first(sllist *ptr_sll, int value)
    {
        if (!ptr_sll->head)
            return;
        node *prev = ptr_sll->head, *curr = prev->next;
        if (prev->value == value)
        {
            ptr_sll->head = curr;
            delete prev;
            return;
        }
        while (curr->value != value)
        {
            prev = curr;
            curr = curr->next;
            if (!curr)
                return;
        }
        auto temp = curr->next;
        delete curr;
        prev->next = temp;
    }

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
    void insert(sllist *ptr_sll, int value, size_t index)
    {
        Helper::insert(ptr_sll->head, value, index - 1);
    }

    /*!*****************************************************************************
	\brief
        Returns the first node in the linked-list
    \param[in] ptr_sll
        Pointer that stores the head node pointer
    \return
        Pointer to the first node        
    *******************************************************************************/
    node* front(sllist *ptr_sll)
    {
        return ptr_sll->head;
    }

    /*!*****************************************************************************
	\brief
        Return the first node in the linked-list for reference only
    \param[in] ptr_sll
        Pointer that stores the head node pointer
    \return
        Pointer to the first node
    *******************************************************************************/
    node const* front(sllist const *ptr_sll)
    {
        return ptr_sll->head;
    }

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
    node* find(sllist const *ptr_sll, int value)
    {
        node *ptr = ptr_sll->head;
        while (ptr && ptr->value != value)
            ptr = ptr->next;
        return ptr;
    }

}

namespace
{
    /*!*****************************************************************************
	\brief
        Helper function that creates a new node
    \param[in] value
        Integral data to be stored
    \param[in] next
        Pointer to the next node that the newly allocated node should be pointing
    \return
        Pointer to a newly allocated node
    *******************************************************************************/
    hlp2::node* create_node(int value, hlp2::node* next)
    {
        return new hlp2::node{ value, next };
    }
}

namespace Helper
{
    /*!*****************************************************************************
	\brief
        Does a recursive function to add a new node at the back
    \param[in] curr
        Pointer to the current node to be checked
	\param[in] value
        Integral value to be stored
    *******************************************************************************/
    void push_back(hlp2::node*& curr, int value)
    {
        if (!curr)
        {
            curr = create_node(value);
            return;
        }
        push_back(curr->next, value);
    }

    /*!*****************************************************************************
	\brief
        Insert a new node at position specified by index
    \param[in] curr
        Pointer to the next node
	\param[in] value
        Pointer to the current node to be checked
    \param[in] index
        Current position in the linked-list
    *******************************************************************************/
    void insert(hlp2::node*& curr, int value, int index)
    {
        if (0 > index || !curr)
        {
            auto temp = curr;
            curr = create_node(value, temp);
            return;
        }
        insert(curr->next, value, index - 1);
    }
}