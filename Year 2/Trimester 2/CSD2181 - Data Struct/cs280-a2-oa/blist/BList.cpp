#include "BList.h"

namespace
{
  template <typename T>
  void swap(T &lhs, T &rhs)
  {
    T tmp{lhs};
    lhs = rhs;
    rhs = tmp;
  }

  template <typename T>
  void swap(T *lhs, T *rhs)
  {
    T tmp{*lhs};
    *lhs = *rhs;
    *rhs = tmp;
  }
}

template <typename T, unsigned N>
BList<T, N>::BList() : head_(nullptr), tail_(nullptr), isSorted(false), inserted(false)
{
  stats_.NodeSize = nodesize();
  stats_.NodeCount = 0;
  stats_.ArraySize = N;
  stats_.ItemCount = 0;
}

template <typename T, unsigned N>
BList<T, N>::BList(BList const &rhs) : head_(nullptr), tail_(nullptr), isSorted(false), inserted(false)
{
  stats_ = rhs.stats_;
  if (!rhs.head_)
    return;

  // Copy the content of rhs into this BList
  BNode const *it{rhs.head_}; // using it to iterate thru rhs's link list
  BNode *curr{nullptr}, *prev{nullptr};
  curr = head_ = CreateNode();

  // Infinite loop to copy all the node data in rhs into this BList
  while (true)
  {
    for (size_type i = 0; i < N; ++i)
      *(curr->values + i) = *(it->values + i);
    curr->count = it->count;

    it = it->next;
    if (!it)
      break; // If next node is nullptr, break out of the loop

    curr->next = CreateNode(curr, nullptr);
    prev = curr;
    curr = curr->next;
  }
  tail_ = curr;
}

template <typename T, unsigned N>
BList<T, N>::~BList()
{
  clear();
}

template <typename T, unsigned N>
BList<T, N> &BList<T, N>::operator=(const BList &rhs)
{
  BList tmp{rhs};
  Swap(tmp);
  return *this;
}

template <typename T, unsigned N>
void BList<T, N>::push_back(const T &value)
{
}

template <typename T, unsigned N>
void BList<T, N>::push_front(const T &value)
{
}

template <typename T, unsigned N>
void BList<T, N>::insert(const T &value)
{
  // The BList is empty
  if (IsEmpty())
  {
    head_ = tail_ = CreateNode();
    PlaceItem(head_, 0, value);
  }
  // BNode's array can only contain 1 element
  else if (N == 1)
    ArraySizeIsOne(value);
  else
  {
    // Head and tail is pointing to the same address
    HeadTailSame(value);
    // Head and tail is not pointing to the same address
    HeadTailNotSame(value);
  }
  inserted = false;
}

template <typename T, unsigned N>
void BList<T, N>::remove(int index)
{
  BNode *node = GetNodeByIndex(index);
  --node->count;
  for (int i{index}; i < node->count; ++i)
    *(node->values + i) = *(node->values + i + 1);
  if (node->count)
    return;
  // current node is empty, remove this node
  if (node->prev)
    node->prev->next = node->next;
  if (node->next)
    node->next->prev = node->prev;
}

template <typename T, unsigned N>
void BList<T, N>::remove_by_value(const T &value)
{
}

template <typename T, unsigned N>
int BList<T, N>::find(const T &value) const
{
  return 0;
}

template <typename T, unsigned N>
T &BList<T, N>::operator[](int index)
{
  return const_cast<reference>(const_cast<BList<T, N> const &>(*this)[index]);
}

template <typename T, unsigned N>
const T &BList<T, N>::operator[](int index) const
{
  BNode *node{GetNodeByIndex(index)};
  return node->values[index];
}

template <typename T, unsigned N>
size_t BList<T, N>::size() const
{
  size_t items{};
  BNode *node{head_};
  while (node)
  {
    items += static_cast<size_t>(node->count);
    node = node->next;
  }
  return items;
}

template <typename T, unsigned N>
void BList<T, N>::clear()
{
  BNode *curr{head_};

  while (curr)
  {
    BNode *next = curr->next;
    DeleteNode(curr);
    curr = next;
  }
  head_ = tail_ = nullptr;
}

template <typename T, unsigned N>
size_t BList<T, N>::nodesize(void)
{
  return sizeof(BNode);
}

template <typename T, unsigned N>
const typename BList<T, N>::BNode *BList<T, N>::GetHead() const
{
  return head_;
}

template <typename T, unsigned N>
BListStats BList<T, N>::GetStats() const
{
  return stats_;
}

template <typename T, unsigned N>
bool BList<T, N>::IsEmpty() const
{
  return !head_ && !tail_;
}

template <typename T, unsigned N>
void BList<T, N>::ArraySizeIsOne(value_type const &value)
{
  if (head_ == tail_)
  {
    if (*head_->values < value)
    {
      head_ = CreateNode(nullptr, tail_);
      tail_->prev = head_;
      PlaceItem(head_, 0, value);
    }
    else
    {
      tail_ = CreateNode(head_, nullptr);
      head_->next = tail_;
      PlaceItem(tail_, 0, value);
    }
  }
  else
  {
    // find the node where value is less than node
    BNode *node{head_};
    while (node->next && *node->values < value)
      node = node->next;

    BNode *newNode{nullptr};
    if (*node->values < value)                // this case will only happen when it reaches the end of the BList
      newNode = CreateNode(node, node->next); // node->next is actl nullptr
    else
      newNode = CreateNode(node->prev, node);
    PlaceItem(newNode, 0, value);

    // Rearrange the links
    if (node == head_)
    {
      head_ = newNode;
      node->prev = head_;
    }
    else if (node == tail_)
    {
      tail_ = newNode;
      node->next = tail_;
    }
    else
    { // inserting a new node in between
      node->prev->next = newNode;
      node->prev = newNode;
    }
  }
}

template <typename T, unsigned N>
void BList<T, N>::HeadTailSame(value_type const &value)
{
  if (head_ != tail_ || inserted)
    return;

  if (head_->count == N)
  {
    tail_ = SplitNode(head_, nullptr);

    if (value < *tail_->values)
    { // put in the left node
      PlaceItem(head_, head_->count, value);
      SortArray(head_);
    }
    else
    {
      PlaceItem(tail_, tail_->count, value);
      SortArray(tail_);
    }
  }
  else
  {
    PlaceItem(head_, head_->count, value);
    SortArray(head_);
  }
}

template <typename T, unsigned N>
void BList<T, N>::HeadTailNotSame(value_type const &value)
{
  if (head_ == tail_ || inserted)
    return;

  // Find the node where value is < the first element in that arr
  BNode *node{head_};
  while (node->next && *node->values < value)
    node = node->next;

  /*
    4 Cases when inserting 15 into the list
    1) When both nodes have empty slots : 10 _ -- 20 _
      -> Insert 15 on the left node
      Result: 10 15 -- 20 _

    2) When both nodes are full : 10 12 -- 20 22
      -> Split left node
      Result: 10 _ -- 12 _ -- 20 22
      -> Insert 15 into right node
      Result: 10 _ -- 12 15 -- 20 22
      If the list is 10 18 -- 20 22 instead then
      -> Split left node and insert 15 into the left node
      Result: 10 15 -- 18 _ -- 20 22

    3) When left node is full and right node has slot : 10 12 -- 20 _
      -> Insert 15 into the right node
      Result: 10 12 -- 15 20

    4) When right node is full and left node has slot : 10 _ -- 20 22
      -> Insert 15 into left node
      Result: 10 15 -- 20 22
  */
  if (value < *node->values)
  {
    // Inserting into head node
    if (node == head_ && node->count >= N)
    { // Current node is full
      head_ = SplitNode(node->prev, node);
      if (value < *head_->values)
      {
        PlaceItem(head_, head_->count, value);
        SortArray(head_);
      }
      else
      {
        PlaceItem(node, node->count, value);
        SortArray(node);
      }
      return;
    }

    BNode *prev = node->prev;
    if (prev->count < N)
    { // Case 1 and 4
      PlaceItem(prev, prev->count, value);
      SortArray(prev);
    }
    else if (prev->count >= N && node->count < N)
    { // Case 3
      PlaceItem(node, node->count, value);
      SortArray(node);
    }
    else if (prev->count >= N && node->count >= N)
    { // case 2
      node = SplitNode(prev, node);
      if (value < *node->values)
      {
        PlaceItem(prev, prev->count, value);
        SortArray(prev);
      }
      else
      {
        PlaceItem(node, node->count, value);
        SortArray(node);
      }
    }
  }
  else
  { // If the value to be inserted is greater than the first element of the node's array, this node is definitely the tail
    if (node->count >= N)
    {
      tail_ = SplitNode(node, node->next);
      if (value < *tail_->values)
      {
        PlaceItem(node, node->count, value);
        SortArray(node);
      }
      else
      {
        PlaceItem(tail_, tail_->count, value);
        SortArray(tail_);
      }
    }
    else
    {
      PlaceItem(node, node->count, value);
      SortArray(node);
    }
  }
}

template <typename T, unsigned N>
void BList<T, N>::PlaceItem(BNode *node, size_t pos, value_type const &value)
{
  *(node->values + pos) = value;
  ++node->count;
  ++stats_.ItemCount;
  inserted = true;
}

template <typename T, unsigned N>
void BList<T, N>::SortArray(BNode *node)
{
  if (!node->count)
    return;

  T *arr = node->values;

  for (size_type i = 0; i < static_cast<size_type>(node->count - 1); ++i)
  {
    size_type min = i, j{};
    for (j = i + 1; j < N; ++j)
      if (arr[j] < arr[min])
        min = j;
    swap(arr + min, arr + i);
  }
}

template <typename T, unsigned N>
typename BList<T, N>::BNode *BList<T, N>::CreateNode(BNode *prev, BNode *next)
{
  BNode *ptr{nullptr};

  try
  {
    ptr = new BNode{};
    ++stats_.NodeCount;
  }
  catch (std::bad_alloc const &e)
  {
    throw BListException(BListException::BLIST_EXCEPTION::E_NO_MEMORY, e.what());
  }

  ptr->prev = prev;
  ptr->next = next;
  return ptr;
}

template <typename T, unsigned N>
typename BList<T, N>::BNode *BList<T, N>::SplitNode(BNode *prev, BNode *next)
{
  BNode *ptr = CreateNode(prev, next);
  if (next)
    next->prev = ptr;
  if (prev)
    prev->next = ptr;

  prev->count = ptr->count = N >> 0b1;
  for (size_type i = 0, j = prev->count; i < static_cast<size_type>(prev->count); ++i, ++j)
    *(ptr->values + i) = *(prev->values + j);

  return ptr;
}

template <typename T, unsigned N>
void BList<T, N>::DeleteNode(BNode *&ptr)
{
  delete ptr;
  ptr = nullptr;
  --stats_.NodeCount;
}

template <typename T, unsigned N>
typename BList<T, N>::BNode *BList<T, N>::GetNodeByIndex(int &index) const
{
  int const tmp{index};
  if ((stats_.ArraySize * stats_.NodeCount) <= tmp)
    throw BListException(BListException::BLIST_EXCEPTION::E_BAD_INDEX, "Invalid index: " + tmp);

  BNode *node{head_};
  while (node->next)
  {
    if (index < node->count)
      break;
    index -= node->count;
    node = node->next;
  }

  if (index >= node->count)
    throw BListException(BListException::BLIST_EXCEPTION::E_BAD_INDEX, "Invalid index: " + tmp);
  return node;
}

template <typename T, unsigned N>
void BList<T, N>::Swap(BList &tmp)
{
  swap(head_, tmp.head_);
  swap(tail_, tmp.tail_);
  swap(stats_, tmp.stats_);
}