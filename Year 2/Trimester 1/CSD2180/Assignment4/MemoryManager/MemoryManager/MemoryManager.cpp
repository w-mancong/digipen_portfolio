#include "MemoryManager.h"

MemoryManager::MemoryManager(int total_bytes)
{
	m_Memory = reinterpret_cast<char*>( malloc(static_cast<size_t>(total_bytes)) );
	m_TotalBytes = total_bytes;
}

MemoryManager::~MemoryManager(void)
{
	free(m_Memory);
}

void* MemoryManager::allocate(int bytes)
{
	if (bytes + m_AllocatedBytes > m_TotalBytes)
	{
		std::cerr << "Error: Unable to allocate memory. Memory block not enough to allocate given size " << bytes << std::endl << std::endl;
		return nullptr;
	}

	uint64_t nextIndex = 0;
	std::list<Node>::iterator it{ m_NodeList.begin() };
	bool insertIntoList{ true };
	if (!m_NodeList.empty())
	{	
		for (; it != m_NodeList.end(); ++it)
		{	
			// There is a chunk of memory available in the middle of the list		
			if (!it->allocated)
				break;
		}
	
		bool notEoughBytes = { false };
		if (it != m_NodeList.end())
		{	// There is a chunk of memory in the middle of the list
			notEoughBytes = it->byteCount < bytes;
			if (it->byteCount == bytes)
			{
				nextIndex = static_cast<uint64_t>(it->startAddress - m_Memory);
				it->allocated = true;
				insertIntoList = false;
			}
			else if (!notEoughBytes)
			{	// This node data has enough memory to be allocated
				nextIndex = static_cast<uint64_t>(it->startAddress - m_Memory);
				it->byteCount -= bytes;
				it->startAddress += bytes;
			}
		}
		if (nextIndex == 0)
		{		
			Node const& node =  m_NodeList.back();
			nextIndex = static_cast<uint64_t>((node.startAddress + node.byteCount) - m_Memory);

			if (nextIndex + bytes > m_TotalBytes)
			{
				std::cerr << "Error: Unable to allocate memory. Memory block not enough to allocate given size " << bytes << std::endl << std::endl;
				return nullptr;
			}
			if(notEoughBytes)
				it = m_NodeList.end();
		}
	}

	Node data{ m_Memory + nextIndex, bytes, true };
	if(insertIntoList)
		m_NodeList.insert(it, data);
	m_AllocatedBytes += bytes;

	return data.startAddress;
}

void MemoryManager::deallocate(void *ptr)
{
	if (!m_NodeList.empty())
	{
		// 'it' will contain the iteration when ptr matches any startAddress in m_NodeList
		bool found{ false };
		std::list<Node>::iterator it{ m_NodeList.begin() };
		for (; it != m_NodeList.end(); ++it)
		{	// Loop through m_NodeList to find ptr of startAddress
			if (it->startAddress != ptr)
				continue;

			found = true;
			it->allocated = false;
			m_AllocatedBytes -= it->byteCount;
			break;
		}	

		if (found)
		{
			// it == m_NodeList's last iterator, means can "combine" with the rest of the memory pool
			if (it->startAddress == m_NodeList.back().startAddress)
			{
				it->byteCount = 0;
				it->startAddress = nullptr;
				m_NodeList.erase(it);
				return;
			}

			// if found a node in m_NodeList, loop through m_NodeList again to find any free nodes and combine them into one big chunk
			for (std::list<Node>::iterator it2{ m_NodeList.begin() }; it2 != m_NodeList.end(); ++it2)
			{
				if (it2->allocated || it == it2)
					continue;

				if ((it->startAddress + it->byteCount) == it2->startAddress)
				{	// 'it' comes before it2
					it->byteCount += it2->byteCount;
					m_NodeList.erase(it2);
				}
				else if ((it2->startAddress + it2->byteCount) == it->startAddress)
				{	// it2 comes before 'it'
					it2->byteCount += it->byteCount;
					m_NodeList.erase(it);
				}
				break;
			}
			return;
		}
	}
	// m_NodeList is empty or cannot find ptr within m_NodeList
	std::cerr << "Error: Unable to find memory address." << std::endl << std::endl;
}

void MemoryManager::dump(std::ostream &out)
{
	if (m_NodeList.empty())
		Print(reinterpret_cast<void*>(m_Memory), m_TotalBytes, false, out);
	for (Node const& data : m_NodeList)
		Print(reinterpret_cast<void*>(data.startAddress), data.byteCount, data.allocated, out);
}

void MemoryManager::Print(void* startAddress, int byteCount, bool allocated, std::ostream& out)
{
	out << "start address: " << startAddress << std::endl;
	out << "byte count"		 << std::setw(5) << ": " << std::hex << byteCount << "h" << std::endl;
	out << "allocated?"		 << std::setw(5) << ": " << std::boolalpha << allocated << std::endl;
}