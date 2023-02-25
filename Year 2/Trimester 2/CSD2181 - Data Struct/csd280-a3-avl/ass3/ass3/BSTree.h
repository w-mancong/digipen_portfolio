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

/*!
  The definition of the BST
*/
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

	BSTree(ObjectAllocator* oa = nullptr, bool ShareOA = false);
	BSTree(const BSTree& rhs);
	virtual ~BSTree();
	BSTree& operator=(const BSTree& rhs);
	const BinTreeNode* operator[](int index) const; // for r-values (Extra Credit)
	virtual void insert(const T& value);
	virtual void remove(const T& value);
	void clear();
	bool find(const T& value, unsigned& compares) const;
	bool empty() const;
	unsigned int size() const;
	int height() const;
	BinTree root() const;

protected:
	BinTree& get_root();
	BinTree make_node(const T& value) const;
	void free_node(BinTree node);
	int tree_height(BinTree tree) const;
	void find_predecessor(BinTree tree, BinTree& predecessor) const;
	BinTree find(BinTree node, T const& value) const;

private:
	unsigned int size(BinTree node) const;
	bool find(BinTree node, T const& value, unsigned& compares) const;
	BinTree insert(BinTree node, T const& value);
	BinTree remove(BinTree node, T const& value);
	int Height(BinTree node, int height) const;
	int CalculateCount(BinTree node) const;
	void swap(BSTree& tmp);
	void CopyTree(BinTree node);
	void ClearTree(BinTree node);

	ObjectAllocator* m_pOA{ nullptr };

	BinTree m_pRoot{ nullptr };

	bool m_ShareOA{ false };
	bool m_FreeOA{ false };
};

//#include "BSTree.cpp"

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
BSTree<T>::BSTree(ObjectAllocator* oa, bool ShareOA) : m_pOA{ oa }, m_ShareOA{ ShareOA }, m_FreeOA{ ShareOA }
{
	if (!oa)
	{
		OAConfig config{ true };
		m_pOA = new ObjectAllocator(sizeof(BinTreeNode), config);
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
	return nullptr;
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
	return static_cast<const BSTree<T>>(*this).root();
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
unsigned int BSTree<T>::size(BinTree node) const
{
	if (!node) return 0;
	return size(node->left) + 1 + size(node->right);
}

template <typename T>
bool BSTree<T>::find(BinTree node, T const& value, unsigned& compares) const
{
	++compares;
	if (!node) return false;
	if (value < node->data) // go to left subtree
		find(node->left, value, compares);
	else
		find(node->right, value, compares);
	return true;
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
int BSTree<T>::CalculateCount(BinTree node) const
{
	if (!node) return 1;
	node->count = CalculateCount(node->left) + CalculateCount(node->right) - 1;
	return node->count + 1;
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
	remove(node->data);
}

#endif
//---------------------------------------------------------------------------
