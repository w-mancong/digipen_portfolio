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
		BinTreeNode() : left(0), right(0), data(0), balance_factor(0), count(1) {};

		//! Conversion constructor
		BinTreeNode(const T& value) : left(0), right(0), data(value), balance_factor(0), count(1) {};
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

private:
	int Height(BinTree node, int height) const;
	void swap(BSTree& tmp);
	
	ObjectAllocator* m_pOA{ nullptr };

	BinTree m_pRoot{ nullptr };

	unsigned int m_totalNodes{ 0 };

	bool m_ShareOA{ false };
	bool m_FreeOA{ false };
};

//#include "BSTree.cpp"

namespace
{
	template <typename T>
	T max(T a, T b)
	{
		return  a < b  ?  b : a;
	}

	template <typename T>
	void swap(T& lhs, T& rhs)
	{
		T tmp{ lhs };
		lhs = rhs;
		rhs = lhs;
	}

	template <typename T>
	void swap(T*& lhs, T*& rhs)
	{
		T* tmp{ lhs };
		lhs = rhs;
		rhs = lhs;
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
BSTree<T>::BSTree(BSTree const& rhs) : m_totalNodes{ rhs.m_totalNodes }
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


}

template <typename T>
BSTree<T>::~BSTree(void)
{
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

}

template <typename T>
void BSTree<T>::remove(T const& value)
{

}

template <typename T>
void BSTree<T>::clear(void)
{

}

template <typename T>
bool BSTree<T>::find(const T& value, unsigned& compares) const
{
	return false;
}

template <typename T>
bool BSTree<T>::empty() const
{
	return !m_pRoot;
}

template <typename T>
unsigned int BSTree<T>::size() const
{
	return m_totalNodes;
}

template <typename T>
int BSTree<T>::height() const
{
	return Height(m_pRoot, 0);
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
		ptr = m_pOA->Allocate();
	}
	catch (OAException const& e)
	{
		throw BSTException(BSTException::BST_EXCEPTION::E_NO_MEMORY, "No more memory!");
	}

	++m_totalNodes;
	return ptr;
}

template <typename T>
void BSTree<T>::free_node(BinTree node)
{
	m_pOA->Free(node);
	--m_totalNodes;
}

template <typename T>
int BSTree<T>::tree_height(BinTree tree) const
{
	return Height(tree, 0);
}

template <typename T>
void BSTree<T>::find_predecessor(BinTree tree, BinTree& predecessor) const
{

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
	::swap(m_pOA, tmp.m_pOA);
	::swap(m_pRoot, tmp.m_pRoot);
	::swap(m_ShareOA, tmp.m_ShareOA);
	::swap(m_FreeOA, tmp.m_FreeOA);
}

#endif
//---------------------------------------------------------------------------
