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
	virtual void insert(const T& value) override;
	virtual void remove(const T& value) override;

	// Returns true if efficiency implemented
	static bool ImplementedBalanceFactor(void);

private:
	using BinTree = typename BSTree<T>::BinTree;

	// private stuff
	int BalanceFactor(BinTree node);
	BinTree insert(BinTree node, T const& value);
	BinTree Balance(BinTree node);
	BinTree LLRotation(BinTree node);
	BinTree LRRotation(BinTree node);
	BinTree RRRotation(BinTree node);
	BinTree RLRotation(BinTree node);
};

//#include "AVLTree.cpp"

template <typename T>
AVLTree<T>::AVLTree(ObjectAllocator* oa, bool ShareOA) : BSTree<T>(oa, ShareOA)
{

}

template <typename T>
void AVLTree<T>::insert(T const& value)
{
	BinTree& root = this->get_root();
	root = insert(root, value);
	root->count = this->CalculateCount(root) - 1;
}

template <typename T>
void AVLTree<T>::remove(T const& value)
{

}

template <typename T>
bool AVLTree<T>::ImplementedBalanceFactor(void)
{
	return true;
}

template<typename T>
int AVLTree<T>::BalanceFactor(BinTree node)
{
	return this->tree_height(node->left) - this->tree_height(node->right);
}

template <typename T>
typename AVLTree<T>::BinTree AVLTree<T>::insert(BinTree node, T const& value)
{
	if (!node)
		return this->make_node(value);
	else if (value < node->data)
		node->left = insert(node->left, value);
	else if (node->data < value)
		node->right = insert(node->right, value);
	node->balance_factor = BalanceFactor(node);
	return Balance(node);
}

template <typename T>
typename AVLTree<T>::BinTree AVLTree<T>::Balance(BinTree node)
{
	if (node->balance_factor == 2)
	{
		if (node->left->balance_factor >= 0)
		{	// Left heavy, do LLRotation
			return LLRotation(node);
		}
		else
		{	// Left-Right Heavy, do LRRotation
			return LRRotation(node);
		}
	}
	else if (node->balance_factor == -2)
	{
		if (node->right->balance_factor <= 0)
		{	// Right heavy, do RRRotation
			return RRRotation(node);
		}
		else
		{	// Right-Left heavy, do RLRotation
			return RLRotation(node);
		}
	}
	return node;
}

template <typename T>
typename AVLTree<T>::BinTree AVLTree<T>::LLRotation(BinTree node)
{
	BinTree pivot{ node };

	node = node->left;
	pivot->left = node->right;
	node->right = pivot;

	return node;
}

template <typename T>
typename AVLTree<T>::BinTree AVLTree<T>::LRRotation(BinTree node)
{
	node->left = RRRotation(node->left);
	return LLRotation(node);
}

template <typename T>
typename AVLTree<T>::BinTree AVLTree<T>::RRRotation(BinTree node)
{
	BinTree pivot{ node };

	node = node->right;
	pivot->right = node->left;
	node->left = pivot;

	return node;
}

template <typename T>
typename AVLTree<T>::BinTree AVLTree<T>::RLRotation(BinTree node)
{
	node->right = LLRotation(node->right);
	return RRRotation(node);
}

#endif
//---------------------------------------------------------------------------
