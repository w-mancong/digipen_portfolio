/*!*****************************************************************************
\file   BSTree.h
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: CSD2181
\par Section: A
\par Assignment 3 - AVL Tree 
\date 27/02/2023
\brief
This file contains functon declaration for defining a binary search tree (BST) 
*******************************************************************************/
//---------------------------------------------------------------------------
#ifndef BSTREE_H
#define BSTREE_H
//---------------------------------------------------------------------------
#include <string>    // std::string
#include <stdexcept> // std::exception

#include "ObjectAllocator.h"

/*!
  The exception class for the AVL/BST classes
*/
class BSTException : public std::exception
{
public:
	/*!
	  Non-default constructor

	  \param ErrCode
		The kind of exception (only one currently)

	  \param Message
		The human-readable reason for the exception.
	*/
	BSTException(int ErrCode, const std::string& Message) :
		error_code_(ErrCode), message_(Message) {
	};

	/*!
	  Retrieve the exception code.

	  \return
		E_NO_MEMORY
	*/
	virtual int code() const {
		return error_code_;
	}

	/*!
	  Retrieve the message string

	  \return
		The human-readable message.
	*/
	virtual const char* what() const throw() {
		return message_.c_str();
	}

	//! Destructor
	virtual ~BSTException() {}

	//! The kinds of exceptions (only one currently)
	enum BST_EXCEPTION { E_NO_MEMORY };

private:
	int error_code_;      //!< The code of the exception
	std::string message_; //!< Readable message text
};

/*!*****************************************************************************
    \brief The definition of the BST
*******************************************************************************/
template <typename T>
class BSTree
{
public:
	//! The node structure
	struct BinTreeNode
	{
		BinTreeNode* left;  //!< The left child
		BinTreeNode* right; //!< The right child
		T data;             //!< The data
		int balance_factor; //!< optional for efficient balancing
		unsigned count;     //!< nodes in this subtree for efficient indexing

		//! Default constructor
		BinTreeNode() : left(nullptr), right(nullptr), data(nullptr), balance_factor(0), count(1) {};

		//! Conversion constructor
		BinTreeNode(const T& value) : left(nullptr), right(nullptr), data(value), balance_factor(0), count(1) {};
	};

	//! shorthand
	using BinTree = BinTreeNode*;

    /*!*****************************************************************************
        \brief Default constructor for BSTree

        \param [in] oa: Pointer to an ObjectAllocator. If this parameter is null,
        an ObjectAllocator will be instantiated
        \param [in] ShareOA: Used to determine if a BSTree should share an ObjectAllocator
        when copying a BSTree. If false, an ObjectAllocator will be instantiated
    *******************************************************************************/
	BSTree(ObjectAllocator* oa = nullptr, bool ShareOA = false);
    
    /*!*****************************************************************************
        \brief Copy Constructor
    *******************************************************************************/
	BSTree(const BSTree& rhs);
    
    /*!*****************************************************************************
        \brief Destructor
    *******************************************************************************/
	virtual ~BSTree();
    
    /*!*****************************************************************************
        \brief Copy Assignment
    *******************************************************************************/
	BSTree& operator=(const BSTree& rhs);
    
    /*!*****************************************************************************
        \brief Overloaded subscript operator to access BSTree by index
    *******************************************************************************/
	const BinTreeNode* operator[](int index) const; // for r-values (Extra Credit)
    
    /*!*****************************************************************************
        \brief Insert value into BSTree

        \param [in] value: Value to be inserted into the BSTree

        \exception BST_EXCEPTION::E_NO_MEMORY will be thrown if ObjectAllocator fails
        to allocate
    *******************************************************************************/
	virtual void insert(T const& value);
    
    /*!*****************************************************************************
        \brief Remove value from BSTree

        \param [in] value: Value to be removed from the BSTree
    *******************************************************************************/
	virtual void remove(T const& value);
    
    /*!*****************************************************************************
        \brief Clear the tree
    *******************************************************************************/
	void clear();
    
    /*!*****************************************************************************
        \brief Find a particular value in the BSTree

        \param [in] value: Value to find
        \param [in, out] compares: Number of comparison before value is found/not found

        \return True if value is found inside BSTree, else false
    *******************************************************************************/
	bool find(const T& value, unsigned& compares) const;
    
    /*!*****************************************************************************
        \brief Check if the BSTree is empty

        \return True if empty, else false
    *******************************************************************************/
	bool empty() const;
    
    /*!*****************************************************************************
        \brief Return the total number of nodes in BSTree

        \return Size of BSTree
    *******************************************************************************/
	unsigned int size() const;
 
    /*!*****************************************************************************
        \brief Returns the height of the BSTree from the root
    *******************************************************************************/
	int height() const;
    
    /*!*****************************************************************************
        \brief Returns the root to BSTree
    *******************************************************************************/
	BinTree root() const;

protected:
    /*!*****************************************************************************
        \brief Returns a reference to the root
    *******************************************************************************/
	BinTree& get_root();
    
    /*!*****************************************************************************
        \brief Helper function to create a node
    *******************************************************************************/
	BinTree make_node(const T& value) const;
    
    /*!*****************************************************************************
        \brief Helper function to free a node
    *******************************************************************************/
	void free_node(BinTree node);
    
    /*!*****************************************************************************
        \brief Returns the height of BSTree from a particular node
    *******************************************************************************/
	int tree_height(BinTree tree) const;
    
    /*!*****************************************************************************
        \brief Finds the predecessor of a particular node. The predecessor of the node
        will be the right most leaf from it's left subtree 

        \param [in] node: To find the predeccessor of this particular node
        \param [in, out] predeccessor: Once the predecessor of node is found, 
        store it into this variable
    *******************************************************************************/
	void find_predecessor(BinTree node, BinTree& predecessor) const;
        
    /*!*****************************************************************************
        \brief Recursive function to calculate the count of a particular node

        \return The total count for the node
    *******************************************************************************/
	int CalculateCount(BinTree node) const;

private:        
    /*!*****************************************************************************
        \brief Recursive function to calculate the total number of nodes in BSTree

        \param [in] node: Used for the recursive function to iterate through the
        left and right node

        \return Size of the BSTree
    *******************************************************************************/
	unsigned int size(BinTree node) const;
    
    /*!*****************************************************************************
        \brief  Recursive function to find a particular node

        \param [in] node: Used for the recursive function to iterate through the
        left and right node
        \param [in] value: Value to find in the BSTree

        \return Pointer to the node that contains the value, else nullptr
    *******************************************************************************/
	BinTree find(BinTree node, T const& value) const;
            
    /*!*****************************************************************************
        \brief Resursive function to find a particular node

        \param [in] node: Used for the recursive function to iterate through the 
        left and right node
        \param [in] value: Value to find in the BSTree
        \param [in] compares: Number of comparison before value is found/not found

        \return True if value is found, else false
    *******************************************************************************/
	bool find(BinTree node, T const& value, unsigned& compares) const;
            
    /*!*****************************************************************************
        \brief Recursive function to insert a node into BSTree

        \param [in] node: Used for the recursive function to iterate through the
        left and right node
        \param [in] value: Value to insert

        \return Pointer to the newly inserted node
    *******************************************************************************/
	BinTree insert(BinTree node, T const& value);
            
    /*!*****************************************************************************
        \brief Recursive function to remove a node from BSTree

        \param [in] node: Used for the recursive function to iterate through the
        left and right node
        \param [in] value: Value to remove

        \return Pointer to the rearranged node after removal of value from BSTree 
    *******************************************************************************/
	BinTree remove(BinTree node, T const& value);
            
    /*!*****************************************************************************
        \brief Recursive function to find the height of BSTree from a particular node

        \param [in] node: Used for the recursive function to iterate through the
        left and right node
        \param [in] height: height of tree at node

        \return Height of tree starting from node
    *******************************************************************************/
	int Height(BinTree node, int height) const;
            
    /*!*****************************************************************************
        \brief Helper function to swap the values of BSTree. Used inside copy assignment
    *******************************************************************************/
	void swap(BSTree& tmp);
            
    /*!*****************************************************************************
        \brief Helper function to copy nodes recursively. Used inside copy constructor
    *******************************************************************************/
	void CopyTree(BinTree node);
            
    /*!*****************************************************************************
        \brief Helper function to remove node recursively. Used inside the clear() function
    *******************************************************************************/
	void ClearTree(BinTree node);
            
    /*!*****************************************************************************
        \brief Helper function to retrieve a node by index recursively. Used inside 
        the overloaded subscript operator[]. 
    *******************************************************************************/
	BinTree GetBinTreeAtIndex(BinTree node, int index) const;

	ObjectAllocator* m_pOA{ nullptr };

	BinTree m_pRoot{ nullptr };

	bool m_ShareOA{ false };
	bool m_FreeOA{ false };
};

#include "BSTree.cpp"

#endif
//---------------------------------------------------------------------------
