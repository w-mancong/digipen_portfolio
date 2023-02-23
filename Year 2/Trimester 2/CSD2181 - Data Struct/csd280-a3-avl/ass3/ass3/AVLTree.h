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
    AVLTree(ObjectAllocator *oa = nullptr, bool ShareOA = false);
    virtual ~AVLTree() = default; // DO NOT IMPLEMENT
    virtual void insert(const T& value) override;
    virtual void remove(const T& value) override;

      // Returns true if efficiency implemented
    static bool ImplementedBalanceFactor(void);

  private:
    // private stuff
};

//#include "AVLTree.cpp"

template <typename T>
AVLTree<T>::AVLTree(ObjectAllocator* oa, bool ShareOA) : BSTree<T>(oa, ShareOA)
{

}

template <typename T>
void AVLTree<T>::insert(T const& value) 
{

}

template <typename T>
void AVLTree<T>::remove(T const& value)
{

}

template <typename T>
bool AVLTree<T>::ImplementedBalanceFactor(void)
{
    return false;
}

#endif
//---------------------------------------------------------------------------
