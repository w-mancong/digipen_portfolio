/*!*****************************************************************************
\file allocator.hpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Assignment 6
\date 6-11-2022
\brief
This file contains function definition that mimics a custom memory allocator
*******************************************************************************/
#include <forward_list>
#include <cstring>

/*!*****************************************************************************
    \brief Overloaded global new operator

    \param [in] size: Total size in bytes to allocate

    \return Memory address of the newly allocated address
*******************************************************************************/
auto operator new(size_t size) -> void*
{
  std::cout << "  Global allocate " << size << " bytes." << std::endl;
  return malloc(size);
}

/*!*****************************************************************************
    \brief Overloaded global delete operator

    \param [in] p: address to be deallocated
*******************************************************************************/
auto operator delete(void* p) noexcept -> void
{
  std::cout << "  Global deallocate." << std::endl;
  free(p);
}

/*!*****************************************************************************
    \brief Overloaded global delete operator (need this for GCC compiler to run)

    \param [in] p: address to be deallocated
*******************************************************************************/
auto operator delete(void *p, size_t size) noexcept -> void
{
  (void)size;
  ::operator delete(p);
}

namespace csd2125 
{
  template <typename TDataType, typename TFlags>
  class allocator 
  {	
  public:
    using value_type      = TDataType;
    using size_type       = size_t;
    using reference       = value_type &;
    using const_reference = value_type const &;
    using pointer         = value_type *;
    using const_pointer   = value_type const *;
    using iterator        = pointer;
    using const_iterator  = const_pointer;

    /*!*****************************************************************************
        \brief Allocate a pointer from allocator

        \param [in] count: total number of elements to be allocated

        \return A pointer to the first avaliable address in the allocator

        \exception Throwing std::bad_alloc if program request the allocator to 
        allocate count number of TDataType, where count is greater than the number
        of elements in a single pool.
    *******************************************************************************/
    auto allocate(size_type count) -> pointer
    {
      std::cout << std::endl;
      std::cout << "  Allocator allocate " << count << " elements. " << std::endl;
      if(sizeof(TFlags) * 8 < count)
        throw std::bad_alloc();
      // Start allocating if list is empty
      if (!std::distance(list.begin(), list.end()))
        return create_pool(count);

      // Check each pool to see if there is enough continous pool to allocate memory
      size_type constexpr BITS{sizeof(TFlags) * 8};
      for (auto it{list.begin()}; it != list.end(); ++it)
      {
        Pool& pool = (*it);
        size_type index{ 0 }, counter{ 0 };

        for (; index < BITS; ++index)
        {
          if(pool.bits & (0b1 << index))
          {
            counter = 0;
            continue;
          }
          ++counter;
          if(counter == count)
          {
            index -= counter - 1;
            break;
          }
        }

        // found a space for count number of elements
        if(counter == count)
        {
          std::cout << "  Found space in a pool for " << count << " elements at index " << index << "." << std::endl;

          size_type const TOTAL_BITS = index + count;
          for(size_type i{index}; i < TOTAL_BITS; ++i)
            pool.bits |= (0b1 << i);
          
          return pool.ptr + index;
        }

        std::cout << "  Did not find space in a pool." << std::endl;
        std::cout << "  Checking next available pool..." << std::endl;
      }

      // If it reaches here, means list no more liao and need to allocate another pool
      return create_pool(count);
    }

    /*!*****************************************************************************
        \brief Deallocate pointer from allocator

        \param [in] p: pointer to the address that is potentially allocated by allocator
        \param [in] count: total number of elements to be deallocated

        \exception Throwing std::bad_alloc if program attempts to deallocate a 
        memory address that the allocator does not contain
    *******************************************************************************/
    auto deallocate(pointer p, size_type count) -> void
    {
      std::cout << "  Allocator deallocate " << count << " elements. " << std::endl;

      size_type constexpr BITS{sizeof(TFlags) * 8};
      for (auto it{list.begin()}; it != list.end(); ++it)
      {
        Pool &pool = (*it);
        size_type index{ 0 };
        bool address_found{ false };
        for (; index < BITS; ++index)
        {
          if(pool.ptr + index == p)
          {
            address_found = true;
            break;
          }
        }

        if(address_found)
        {
          std::cout << "  Found " << count << " elements in a pool." << std::endl;

          // Removing the bits
          size_type const TOTAL_BITS = index + count;
          for (size_type i{index}; i < TOTAL_BITS; ++i)
            pool.bits &= ~(0b1 << i);
          if(pool.bits == 0)
          {
            std::cout << "  Removing an empty pool." << std::endl;
            list.remove(pool);
          }
          return;
        }

        std::cout << "  Checking next existing pool..." << std::endl;
      }
      throw std::bad_alloc();
    }

  private:
    struct Pool
    {
      value_type ptr[sizeof(TFlags) * 8]{};
      TFlags bits{ 0 };

      /*!*****************************************************************************
          \brief Used to check if this object is the same as rhs

          \param [in] rhs: Object to be check with this object

          \return true if rhs and this object are the same object, else false
      *******************************************************************************/
      auto operator==(Pool const& rhs) -> bool
      {
        return ptr == rhs.ptr;
      }
    };

    /*!*****************************************************************************
        \brief Helper function to create a new memory pool

        \param [in] count: Total number of element counts

        \return pointer to the first element of the array
    *******************************************************************************/
    auto create_pool(size_type count) -> pointer
    {
      std::cout << "  Allocating a new pool." << std::endl;
      list.emplace_front(Pool{});

      Pool &pool = list.front();
      memset(pool.ptr, 0, sizeof(pool.ptr));

      for(size_type i{}; i < count; ++i)
        pool.bits |= (0b1 << i);

      std::cout << "  Found space in a pool for " << count << " elements at index 0." << std::endl;
      return pool.ptr;
    }
    
    std::forward_list<Pool> list;
  };

  struct vector 
  {
    float x, y, z, w;

    /*!*****************************************************************************
        \brief Default constructor
    *******************************************************************************/
    vector() : x{0}, y{0}, z{0}, w{0} {}

    /*!*****************************************************************************
        \brief Constructor to initialize member variables
    *******************************************************************************/
    vector(float ax, float ay, float az, float aw) : x{ax}, y{ay}, z{az}, w{aw} {}

    /*!*****************************************************************************
        \brief Default destructor
    *******************************************************************************/
    ~vector(void) = default;

    /*!*****************************************************************************
        \brief Overloaded member function new operator
    *******************************************************************************/
    auto operator new(size_t size) -> void*
    {
      std::cout << "  In-class allocate " << size << " bytes." << std::endl;
      return ::operator new (size);
    }

    /*!*****************************************************************************
        \brief Overloaded member function delete operator
    *******************************************************************************/
    auto operator delete(void* p) noexcept -> void
    {
      std::cout << "  In-class deallocate." << std::endl;
      ::operator delete(p);
    }
  }; 

  union vertex 
  {
    vector vertexCoordinates;
    float axisCoordinates[sizeof(vector) / sizeof(float)];

    /*!*****************************************************************************
        \brief Default constructor
    *******************************************************************************/
    vertex() : vertexCoordinates{} {}

    /*!*****************************************************************************
        \brief Constructor to initialize member variables
    *******************************************************************************/
    vertex(float ax, float ay, float az, float aw) : vertexCoordinates{ax, ay, az, aw} { }

    /*!*****************************************************************************
        \brief Default destructor
    *******************************************************************************/
    ~vertex(void) = default;

    /*!*****************************************************************************
        \brief Overloaded new placement new operator
    *******************************************************************************/
    auto operator new(size_t size, void *p) noexcept -> void *
    {
      (void)size;
      return p;
    }
  };

} // end namespace