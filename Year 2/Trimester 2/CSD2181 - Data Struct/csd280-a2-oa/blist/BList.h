/*!
file:   BList.h
author:	Wong Man Cong
email:	w.mancong\@digipen.edu
brief:  This file contains function declaration of a templated BList data struct

        All content © 2022 DigiPen Institute of Technology Singapore. All rights reserved.
*/
/*__________________________________________________________________________________*/
////////////////////////////////////////////////////////////////////////////////
#ifndef BLIST_H
#define BLIST_H
////////////////////////////////////////////////////////////////////////////////

#include <string> // error strings

/*!
  The exception class for BList
*/
class BListException : public std::exception
{
private:
  int m_ErrCode;             //!< One of E_NO_MEMORY, E_BAD_INDEX, E_DATA_ERROR
  std::string m_Description; //!< Description of the exception

public:
  /*!
    Constructor

    \param ErrCode
      The error code for the exception.

    \param Description
      The description of the exception.
  */
  BListException(int ErrCode, const std::string &Description) : m_ErrCode(ErrCode), m_Description(Description){};

  /*!
    Get the kind of exception

    \return
      One of E_NO_MEMORY, E_BAD_INDEX, E_DATA_ERROR
  */
  virtual int code() const
  {
    return m_ErrCode;
  }

  /*!
    Get the human-readable text for the exception

    \return
      The description of the exception
  */
  virtual const char *what() const throw()
  {
    return m_Description.c_str();
  }

  /*!
    Destructor is "implemented" because it needs to be virtual
  */
  virtual ~BListException()
  {
  }

  //! The reason for the exception
  enum BLIST_EXCEPTION
  {
    E_NO_MEMORY,
    E_BAD_INDEX,
    E_DATA_ERROR
  };
};

/*!
  Statistics about the BList
*/
struct BListStats
{
  //!< Default constructor
  BListStats() : NodeSize(0), NodeCount(0), ArraySize(0), ItemCount(0){};

  /*!
    Non-default constructor

    \param nsize
      Size of the node

    \param ncount
      Number of nodes in the list

    \param asize
      Number of elements in each node (array)

    \param count
      Number of items in the list

  */
  BListStats(size_t nsize, int ncount, int asize, int count) : NodeSize(nsize), NodeCount(ncount), ArraySize(asize), ItemCount(count){};

  size_t NodeSize; //!< Size of a node (via sizeof)
  int NodeCount;   //!< Number of nodes in the list
  int ArraySize;   //!< Max number of items in each node
  int ItemCount;   //!< Number of items in the entire list
};

/*!
  The BList class
*/
template <typename T, unsigned N = 1>
class BList
{
public:
  using value_type = T;
  using reference = value_type &;
  using const_reference = value_type const &;
  using pointer = value_type *;
  using const_pointer = value_type const *;
  using size_type = unsigned;

  /*!
    Node struct for the BList
  */
  struct BNode
  {
    BNode *next; //!< pointer to next BNode
    BNode *prev; //!< pointer to previous BNode
    int count;   //!< number of items currently in the node
    T values[N]; //!< array of items in the node

    //!< Default constructor
    BNode() : next(nullptr), prev(nullptr), count(0)
    {
      unsigned char *p = reinterpret_cast<unsigned char *>(values);
      size_t len = sizeof(values);
      while (len)
      {
        (*p++) = '\0';
        --len;
      }
    }
  };

  /*!*********************************************************************************
    \brief Default constructor
  ***********************************************************************************/
  BList();                            

  /*!*********************************************************************************
    \brief Copy constructor
  ***********************************************************************************/
  BList(const BList &rhs);            

  /*!*********************************************************************************
    \brief Destructor
  ***********************************************************************************/
  ~BList();                           

  /*!*********************************************************************************
    \brief Copy assignment operator
  ***********************************************************************************/
  BList &operator=(const BList &rhs); 

  /*!*********************************************************************************
    \brief Arrays are unsorted, push value to the back of the list

    \param [in] value: value to be inserted into the list
  ***********************************************************************************/
  void push_back(const T &value);

  /*!*********************************************************************************
    \brief Arrays are unsorted, push value to the front of the list

    \param [in] value: value to be inserted into the list
  ***********************************************************************************/
  void push_front(const T &value);

  /*!*********************************************************************************
    \brief Arrays will be sorted, insert the value into the list according to its order

    \param [in] value: value to be inserted
  ***********************************************************************************/
  void insert(T const &value);

  /*!*********************************************************************************
    \brief Removing the element at the specified index

    \param [in] index: index of the element to be removed

    \exception BLIST_EXCEPTION::E_BAD_INDEX will be thrown
  ***********************************************************************************/
  void remove(int index);

  /*!*********************************************************************************
    \brief Removing the element that is equivalent to value

    \param [in] value: value to be removed from the list
  ***********************************************************************************/
  void remove_by_value(const T &value);

  /*!*********************************************************************************
    \brief Find the index of value in the list

    \param [in] value: value of the index to find
    
    \return index where the value is found, else -1 if value is not in the list
  ***********************************************************************************/
  int find(const T &value) const; // returns index, -1 if not found

  /*!*********************************************************************************
    \brief Overloaded subscript operator. This is both a accessor and mutator

    \param [in] index: index of the element to retrieve

    \return value at the specified index

    \exception BLIST_EXCEPTION::E_BAD_INDEX will be thrown
  ***********************************************************************************/
  T &operator[](int index);             

  /*!*********************************************************************************
    \brief Overloaded subscript operator. This is an accessor only

    \param [in] index: index of the element to retrieve

    \return value at the specified index

    \exception BLIST_EXCEPTION::E_BAD_INDEX will be thrown
  ***********************************************************************************/
  const T &operator[](int index) const; 

  /*!*********************************************************************************
    \brief Return the total number of elements in the list
  ***********************************************************************************/
  size_t size() const;

  /*!*********************************************************************************
    \brief Delete all nodes
  ***********************************************************************************/
  void clear();

  /*!*********************************************************************************
    \brief Return the size of each BNode
  ***********************************************************************************/
  static size_t nodesize();

  /*!*********************************************************************************
    \brief Returns the pointer to the front of the list
  ***********************************************************************************/
  const BNode *GetHead() const;

  /*!*********************************************************************************
    \brief Returns the stats of this current list
  ***********************************************************************************/
  BListStats GetStats() const;

private:
  BNode *head_{}; //!< points to the first node
  BNode *tail_{}; //!< points to the last node
  bool isSorted{}, inserted{};
  BListStats stats_{};

  /*!*********************************************************************************
    \brief Check if the list is empty
  ***********************************************************************************/
  bool IsEmpty() const;

  /*!*********************************************************************************
    \brief Place value at the specified node and position in the array

    \param [in] node: Node where value will be inserted into
    \param [in] pos: Position of where to place the value into the node's array
    \param [in] value: Value to be placed into the node's array
  ***********************************************************************************/
  void PlaceItem(BNode *node, size_t pos, value_type const &value);

  /*!*********************************************************************************
    \brief Using selection sort to sort the node's array

    \param [in] node: Node where the array will be sorted
  ***********************************************************************************/
  void SortArray(BNode *node);

  /*!*********************************************************************************
    \brief Helper function where it's used inside the insert function. Used only when
           array size of BNode is 1

    \param [in] value: value to be inserted into the list
  ***********************************************************************************/
  void ArraySizeIsOne(value_type const &value);

  /*!*********************************************************************************
    \brief Helper function where it's used inside the insert function. Used only when
           both the head and tail are pointing to the same node

    \param [in] value: value to be inserted into the list
  ***********************************************************************************/
  void HeadTailSame(value_type const &value);

  /*!*********************************************************************************
    \brief Helper function where it's used inside the insert function. Used only when
           both the head and tail are not pointing to the same node. 

    \param [in] value: value to be inserted into the list
  ***********************************************************************************/
  void HeadTailNotSame(value_type const &value);

  /*!*********************************************************************************
    \brief Helper function to create a node

    \param [in] prev: Where the new node prev will point to
    \param [in] next: Where the new node next will point to

    \return Pointer to a new node allocated

    \exception BLIST_EXCEPTION::E_NO_MEMORY will be thrown if new fails to allocate enough space
  ***********************************************************************************/
  BNode *CreateNode(BNode *prev = nullptr, BNode *next = nullptr);

  /*!*********************************************************************************
    \brief Helper function to create a new node and split the values inside prev into two half

    \param [in] prev: Where the new node prev will point to
    \param [in] next: Where the new node next will point to

    \return Pointer to a new node allocated with prev node value are split equally into half

    \exception BLIST_EXCEPTION::E_NO_MEMORY will be thrown if new fails to allocate enough space
  ***********************************************************************************/
  BNode *SplitNode(BNode *prev, BNode *next);

  /*!*********************************************************************************
    \brief Helper function to delete a node and rearrange the link in the list

    \param [in] ptr: Pointer to have it's memory deallocated
  ***********************************************************************************/
  void DeleteNode(BNode *&ptr);

  /*!*********************************************************************************
    \brief Get a node where index is found

    \param [in] index: index where the node resides

    \return Node where the index is found

    \example
    10 _ -- 15 18 -- 20 _ -- 25 27
    If index => 2
    Will return the pointer that points to 15

    If index => 4
    Will return the pointer that points to 25

    \exception BLIST_EXCEPTION::E_BAD_INDEX will be thrown
  ***********************************************************************************/
  BNode *GetNodeByIndex(int &index) const;

  /*!*********************************************************************************
    \brief Helper function to do copy swap idiom
  ***********************************************************************************/
  void Swap(BList &tmp);
};

#include "BList.cpp"

#endif // BLIST_H
