/*!*****************************************************************************
\file   AVLTree.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: CSD2181
\par Section: A
\par Assignment 3 - AVL Tree 
\date 27/02/2023
\brief
This file contains functon definition for AVL Tree
*******************************************************************************/
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
	BinTree& root = this->get_root();
	root = remove(root, value);
	if (root)
		root->count = this->CalculateCount(root) - 1;
}

template <typename T>
bool AVLTree<T>::ImplementedBalanceFactor(void)
{
	return true;
}

template<typename T>
int AVLTree<T>::BalanceFactor(BinTree node)
{
	if (!node) return 0;
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
typename AVLTree<T>::BinTree AVLTree<T>::remove(BinTree node, T const& value)
{
	if (!node) return node;

	if (value < node->data)
		node->left = remove(node->left, value);
	else if (node->data < value)
		node->right = remove(node->right, value);
	else
	{
		BinTree tmp{ node };
		if (!node->left && !node->right)
		{	// This is a leaf node, no children so can just remove
			node = nullptr;
		}
		else if (node->left && !node->right)
		{	// Only the left node has child
			node = node->left;
		}
		else if (!node->left && node->right)
		{	// Only the right node has child
			node = node->right;
		}
		else if (node->left && node->right)
		{	// This node have two children, find it's predeccesor and promote it to be the "root"
			this->find_predecessor(node, tmp);
			node->data = tmp->data;
			node->left = remove(node->left, tmp->data);
			return node;
		}
		this->free_node(tmp);
	}

	if(node)
		node->balance_factor = BalanceFactor(node);
	return Balance(node);
}

template <typename T>
typename AVLTree<T>::BinTree AVLTree<T>::Balance(BinTree node)
{
	if (!node) return nullptr;
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

	if (node)
	{
		pivot->left = node->right;
		node->right = pivot;
	}

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

	if (node)
	{
		pivot->right = node->left;
		node->left = pivot;
	}

	return node;
}

template <typename T>
typename AVLTree<T>::BinTree AVLTree<T>::RLRotation(BinTree node)
{
	node->right = LLRotation(node->right);
	return RRRotation(node);
}