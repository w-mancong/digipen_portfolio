/*!*****************************************************************************
\file   BSTree.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: CSD2181
\par Section: A
\par Assignment 3 - AVL Tree 
\date 27/02/2023
\brief
This file contains functon definition for a binary search tree (BST) 
*******************************************************************************/
namespace
{
	template <typename T>
	T max(T a, T b)
	{
		return  a < b ? b : a;
	}

	template <typename T>
	void swap(T& lhs, T& rhs)
	{
		T tmp{ lhs };
		lhs = rhs;
		rhs = tmp;
	}

	template <typename T>
	void swap(T*& lhs, T*& rhs)
	{
		T* tmp{ lhs };
		lhs = rhs;
		rhs = tmp;
	}
}

template <typename T>
BSTree<T>::BSTree(ObjectAllocator* oa, bool ShareOA) : m_pOA{ oa }, m_ShareOA{ ShareOA }, m_FreeOA{ false }
{
	if (!oa)
	{
		OAConfig config{ true };
		m_pOA = new ObjectAllocator(sizeof(BinTreeNode), config);
		m_FreeOA = true;
	}
}

template <typename T>
BSTree<T>::BSTree(BSTree const& rhs)
{
	if (rhs.m_ShareOA)
	{
		m_pOA = rhs.m_pOA;
		m_FreeOA = false;
		m_ShareOA = true;
	}
	else
	{
		OAConfig config{ true };
		m_pOA = new ObjectAllocator(sizeof(BinTreeNode), config);

		m_FreeOA = true;
		m_ShareOA = false;
	}

	// Do a preorder insert here
	CopyTree(rhs.m_pRoot);
}

template <typename T>
BSTree<T>::~BSTree(void)
{
	clear();
	if (m_FreeOA)
	{
		delete m_pOA;
		m_pOA = nullptr;
	}
}

template <typename T>
BSTree<T>& BSTree<T>::operator=(BSTree const& rhs)
{
	BSTree tmp{ rhs };
	swap(tmp);
	return *this;
}

template <typename T>
typename BSTree<T>::BinTreeNode const* BSTree<T>::operator[](int index) const
{
	return GetBinTreeAtIndex(m_pRoot, index);
}

template <typename T>
void BSTree<T>::insert(T const& value)
{
	m_pRoot = insert(m_pRoot, value);
	m_pRoot->count = CalculateCount(m_pRoot) - 1;
}

template <typename T>
void BSTree<T>::remove(T const& value)
{
	m_pRoot = remove(m_pRoot, value);
	if(m_pRoot)
		m_pRoot->count = CalculateCount(m_pRoot) - 1;
}

template <typename T>
void BSTree<T>::clear(void)
{
	ClearTree(m_pRoot);
}

template <typename T>
bool BSTree<T>::find(const T& value, unsigned& compares) const
{
	return find(m_pRoot, value, compares);
}

template <typename T>
bool BSTree<T>::empty() const
{
	return !m_pRoot;
}

template <typename T>
unsigned int BSTree<T>::size() const
{
	return size(m_pRoot);
}

template <typename T>
int BSTree<T>::height() const
{
	return Height(m_pRoot, -1);
}

template <typename T>
typename BSTree<T>::BinTree BSTree<T>::root() const
{
	return m_pRoot;
}

template <typename T>
typename BSTree<T>::BinTree& BSTree<T>::get_root()
{
	return m_pRoot;
}

template <typename T>
typename BSTree<T>::BinTree BSTree<T>::make_node(T const& value) const
{
	BinTree ptr{ nullptr };

	try
	{
		ptr = reinterpret_cast<BinTree>(m_pOA->Allocate());
		ptr->left = ptr->right = nullptr;
		ptr->data = value;
		ptr->balance_factor = 0;
		ptr->count = 1;
	}
	catch (OAException const& e)
	{
		throw BSTException(BSTException::BST_EXCEPTION::E_NO_MEMORY, "No more memory!");
	}
	return ptr;
}

template <typename T>
void BSTree<T>::free_node(BinTree node)
{
	m_pOA->Free(node);
}

template <typename T>
int BSTree<T>::tree_height(BinTree tree) const
{
	return Height(tree, -1);
}

template <typename T>
void BSTree<T>::find_predecessor(BinTree tree, BinTree& predecessor) const
{
	BinTree leftSubTree = tree->left;
	while (leftSubTree->right) leftSubTree = leftSubTree->right;
	predecessor = leftSubTree;
}

template <typename T>
int BSTree<T>::CalculateCount(BinTree node) const
{
	if (!node) return 1;
	node->count = CalculateCount(node->left) + CalculateCount(node->right) - 1;
	return node->count + 1;
}

template <typename T>
unsigned int BSTree<T>::size(BinTree node) const
{
	if (!node) return 0;
	return size(node->left) + 1 + size(node->right);
}

template <typename T>
typename BSTree<T>::BinTree BSTree<T>::find(BinTree node, T const& value) const
{
	if (!node) return nullptr;
	if (value < node->data) // go to left subtree
		find(node->left, value);
	else if (node->data < value) // go to right subtree
		find(node->right, value);
	return value;
}

template <typename T>
bool BSTree<T>::find(BinTree node, T const& value, unsigned& compares) const
{
	++compares;
	if (!node) 
		return false;
	if (value < node->data) // go to left subtree
		return find(node->left, value, compares);
	else if (node->data < value) // go to right subtree
		return find(node->right, value, compares);
	else return true;
}

template <typename T>
typename BSTree<T>::BinTree BSTree<T>::insert(BinTree node, T const& value)
{
	if (!node) return make_node(value);
	else if (value < node->data)
		node->left = insert(node->left, value);
	else
		node->right = insert(node->right, value);
	return node;
}

template <typename T>
typename BSTree<T>::BinTree BSTree<T>::remove(BinTree node, T const& value)
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
			find_predecessor(node, tmp);
			node->data = tmp->data;
			node->left = remove(node->left, tmp->data);
			return node;
		}
		free_node(tmp);
	}

	return node;
}

template<typename T>
int BSTree<T>::Height(BinTree node, int height) const
{
	if (!node) return height;
	return max(Height(node->left, height + 1), Height(node->right, height + 1));
}

template <typename T>
void BSTree<T>::swap(BSTree& tmp)
{
	::swap(m_pOA,	  tmp.m_pOA);
	::swap(m_pRoot,	  tmp.m_pRoot);
	::swap(m_ShareOA, tmp.m_ShareOA);
	::swap(m_FreeOA,  tmp.m_FreeOA);
}

template <typename T>
void BSTree<T>::CopyTree(BinTree node)
{
	if (!node) return;
	insert(node->data);
	CopyTree(node->left );
	CopyTree(node->right);
}

template <typename T>
void BSTree<T>::ClearTree(BinTree node)
{
	if (!node) return;
	ClearTree(node->left);
	ClearTree(node->right);
	BSTree<T>::remove(node->data);
}

template <typename T>
typename BSTree<T>::BinTree BSTree<T>::GetBinTreeAtIndex(BinTree node, int index) const
{
	if (!node) return nullptr;
	int const L = node->left ? node->left->count : 0;
	if (L > index)
		return GetBinTreeAtIndex(node->left, index);
	else if (L < index)
		return GetBinTreeAtIndex(node->right, index - L - 1);
	else
		return node;
}