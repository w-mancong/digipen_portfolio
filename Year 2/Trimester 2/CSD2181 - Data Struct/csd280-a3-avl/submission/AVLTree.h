/*!*****************************************************************************
\file   AVLTree.h
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: CSD2181
\par Section: A
\par Assignment 3 - AVL Tree 
\date 27/02/2023
\brief
This file contains functon declaration for defining an AVL Tree
*******************************************************************************/
//---------------------------------------------------------------------------
#ifndef AVLTREE_H
#define AVLTREE_H
//---------------------------------------------------------------------------
#include <stack>
#include "BSTree.h"

/*!
  Definition for the AVL Tree
*/
template <typename T>
class AVLTree : public BSTree<T>
{
public:
	AVLTree(ObjectAllocator* oa = nullptr, bool ShareOA = false);
	virtual ~AVLTree() = default; // DO NOT IMPLEMENT
            
    /*!*****************************************************************************
        \brief Insert value into AVLTree

        \param [in] value: Value to be inserted into the AVLTree

        \exception BST_EXCEPTION::E_NO_MEMORY will be thrown if ObjectAllocator fails
        to allocate
    *******************************************************************************/
	virtual void insert(const T& value) override;
            
    /*!*****************************************************************************
        \brief Remove value from AVLTree

        \param [in] value: Value to be removed from the AVLTree
    *******************************************************************************/
	virtual void remove(const T& value) override;

	// Returns true if efficiency implemented
	static bool ImplementedBalanceFactor(void);

private:
	using BinTree = typename BSTree<T>::BinTree;
 
    /*!*****************************************************************************
        \brief Calculate the balance factor (bf) of a particular node

        \param [in] node: Node of the balanace factor to be calculated
        
        \return Balance factor of the node

        \example bf = height of left subtree - height of right subtree
    *******************************************************************************/
	int BalanceFactor(BinTree node);
            
    /*!*****************************************************************************
        \brief Recursive function to insert a node into AVLTree

        \param [in] node: Used for the recursive function to iterate through the
        left and right node
        \param [in] value: Value to be inserted

        \return Pointer to the newly inserted node
    *******************************************************************************/
	BinTree insert(BinTree node, T const& value);
            
    /*!*****************************************************************************
        \brief Recursive function to remove a node from AVLTree

        \param [in] node: Used for the recursive function to iterate through the
        left and right node
        \param [in] value: Value to remove

        \return Pointer to the rearranged node after removal of value from AVLTree
    *******************************************************************************/
	BinTree remove(BinTree node, T const& value);
            
    /*!*****************************************************************************
        \brief To balance a node if the balance factor is not within the range of
        {-1, 0, 1}

        \param [in] node: Node to check if balancing is required
        
        \return The balanced node if balancing is required, else the original node
    *******************************************************************************/
	BinTree Balance(BinTree node);
            
    /*!*****************************************************************************
        \brief Node have a left child and grandchild, do a right rotation
        
        \param [in] node: Node that will be rotated

        \return Pointer to the balanced node

        \example          
        [param] node: 30
        [1]
                        30
                    20
                10
        
        [2]
                    20
                10      30
    *******************************************************************************/
	BinTree LLRotation(BinTree node);
            
    /*!*****************************************************************************
        \brief Node have a left child and a right grandchild, do a left rotation
        about node's left child, then do a right rotation about node

        \param [in] node: node that will be rotated

        \return Pointer to the balanced node

        \example
        [param] node: 30
        [1]
                    30
                10
                    20  

        [2]
                        30
                    20
                10

        [3]
                    20
                10      30
    *******************************************************************************/
	BinTree LRRotation(BinTree node);
            
    /*!*****************************************************************************
        \brief Node have a right child and grandchild, do a left rotation

        \param [in] node: node that will be rotated

        \return Pointer to the balanced node

        \example
        [param] node: 10
        [1]
                10
                    20
                        30

        [2]
                    20
                10      30
    *******************************************************************************/
	BinTree RRRotation(BinTree node);
            
    /*!*****************************************************************************
        \brief Node have a right child and a left grandchild, do a right rotation 
        about node's right child, then do a left rotation about node

        \param [in] node: node that will be rotated

        \return Pointer to the balanced node

        \example
        [param] node: 10
        [1]
                10
                    30
                20
        
        [2]
                10
                    20
                        30

        [3]
                    20
                10      30
    *******************************************************************************/
	BinTree RLRotation(BinTree node);
};

#include "AVLTree.cpp"

#endif
//---------------------------------------------------------------------------