template <typename T>
ChHashTable<T>::ChHashTable(HTConfig const &config, ObjectAllocator *allocator) : m_Config{config}
{
    m_Stats.Allocator_ = allocator;
    m_Stats.TableSize_ = m_Config.InitialTableSize_;
    m_Stats.HashFunc_ = m_Config.HashFunc_;

    if (!allocator)
    {
        m_Stats.Allocator_ = new ObjectAllocator(sizeof(ChHTNode), OAConfig());
        m_OwnAllocator = true;
    }
    m_Head = new HashHeadNode[static_cast<size_t>(m_Stats.TableSize_)];
}

template <typename T>
ChHashTable<T>::~ChHashTable(void)
{
    if (m_OwnAllocator)
    {
        delete m_Stats.Allocator_;
        m_Stats.Allocator_ = nullptr;
    }
    clear();
    delete[] m_Head;
    m_Head = nullptr;
}

template <typename T>
void ChHashTable<T>::insert(char const *key, T const &data)
{
    ExpandTable();
    InsertItem(key, data);
}

template <typename T>
void ChHashTable<T>::remove(char const *key)
{
    RemoveItem(key);
}

template <typename T>
const T &ChHashTable<T>::find(char const *key) const
{
    HashNode node = Search(key);
    if (!node)
        throw HashTableException(HashTableException::HASHTABLE_EXCEPTION::E_ITEM_NOT_FOUND, "Item not found!");
    return node->Data;
}

template <typename T>
void ChHashTable<T>::clear()
{
    size_t const size = static_cast<size_t>(m_Stats.TableSize_);
    for (size_t i{}; i < size; ++i)
    {
        HashHeadNode &head = *(m_Head + i);
        HashNode ptr = head.Nodes;
        while (ptr)
        {
            HashNode tmp{ptr->Next};
            m_Stats.Allocator_->Free(ptr);
            ptr = tmp;
        }
        head.Nodes = nullptr;
    }
    m_Stats.Count_ = 0;
}

template <typename T>
HTStats ChHashTable<T>::GetStats() const
{
    return m_Stats;
}

template <typename T>
typename ChHashTable<T>::ChHTHeadNode const *ChHashTable<T>::GetTable() const
{
    return m_Head;
}

template <typename T>
unsigned ChHashTable<T>::GetIndex(char const *key) const
{
    return m_Stats.HashFunc_(key, m_Stats.TableSize_);
}

template <typename T>
void ChHashTable<T>::ExpandTable(void)
{
    /*
        α = n / m
        where
        α -> load factor
        n -> number of stored keys
        m -> number of slots in hash table
    */
    double load_factor = (m_Stats.Count_ + 1.0) / static_cast<double>(m_Stats.TableSize_);
    if (load_factor <= m_Config.MaxLoadFactor_)
        return;

    struct Temp
    {
        size_t TableSize;
        HashTable head;
    } tmp{static_cast<size_t>(m_Stats.TableSize_), m_Head};

    ++m_Stats.Expansions_;
    // calculating the factor and finding the closest prime number for the new table size
    double factor = std::ceil(m_Stats.TableSize_ * m_Config.GrowthFactor_);
    m_Stats.TableSize_ = GetClosestPrime(static_cast<unsigned>(factor));
    m_Head = new HashHeadNode[m_Stats.TableSize_];

    // inserting old data from previous hash table into new hash table
    for (size_t i{}; i < tmp.TableSize; ++i)
    {
        HashHeadNode const &head = *(tmp.head + i);
        HashNode ptr = head.Nodes;
        while (ptr)
        {
            InsertItem(ptr->Key, ptr->Data, true);
            ptr = ptr->Next;
        }
    }

    { // clearing of previous hash table
        for (size_t i{}; i < tmp.TableSize; ++i)
        {
            HashHeadNode &head = *(tmp.head + i);
            HashNode ptr = head.Nodes;
            while (ptr)
            {
                HashNode tmp{ptr->Next};
                m_Stats.Allocator_->Free(ptr);
                ptr = tmp;
            }
            head.Nodes = nullptr;
        }

        delete[] tmp.head;
        tmp.head = nullptr;
    }
}

template <typename T>
void ChHashTable<T>::InsertItem(char const *key, T const &data, bool reinserting) const
{
    if (Search(key)) // found a duplicate
        throw HashTableException(HashTableException::HASHTABLE_EXCEPTION::E_DUPLICATE, "Duplicated item found!");
    size_t const index = GetIndex(key);
    HashHeadNode &head = *(m_Head + index);

    try
    {
        HashNode newNode = reinterpret_cast<HashNode>(m_Stats.Allocator_->Allocate());
        strcpy(newNode->Key, key);
        newNode->Data = data;

        HashNode prev = head.Nodes;
        newNode->Next = prev;

        head.Nodes = newNode;
        ++head.Count;

        if (!reinserting)
        {
            ++m_Stats.Count_;
        }
    }
    catch (...)
    {
        throw HashTableException(HashTableException::HASHTABLE_EXCEPTION::E_NO_MEMORY, "Out of memory!");
    }
}

template <typename T>
void ChHashTable<T>::RemoveItem(char const *key) const
{
    size_t const index = GetIndex(key);
    HashHeadNode &head = *(m_Head + index);

    HashNode curr = head.Nodes, prev = nullptr;
    while (curr)
    {
        ++m_Stats.Probes_;
        if (!strcmp(curr->Key, key))
        {             // found the item
            if (prev) // removing first item from the list
                prev->Next = curr->Next;
            else
                head.Nodes = curr->Next;
            m_Stats.Allocator_->Free(curr);
            --m_Stats.Count_;
            --head.Count;
            break;
        }
        prev = curr;
        curr = curr->Next;
    }
}

template <typename T>
typename ChHashTable<T>::HashNode ChHashTable<T>::Search(char const *key) const
{
    size_t const index = GetIndex(key);
    HashHeadNode const &head = *(m_Head + index);

    if (head.Count == 0)
    {
        ++m_Stats.Probes_;
        return nullptr;
    }

    HashNode ptr = head.Nodes;
    while (ptr)
    {
        ++m_Stats.Probes_;
        if (!strcmp(ptr->Key, key))
            return ptr;
        ptr = ptr->Next;
    }
    ++m_Stats.Probes_;
    return ptr;
}