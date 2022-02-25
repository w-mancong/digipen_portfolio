/*!*****************************************************************************
\file ocean.cpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Assignment 2
\date 23-01-2022
\brief 
A simple Battleship game that can have the size of the board changed according
to how big the playing field the player decides it to be.
*******************************************************************************/
#include "ocean.h"
#include <iostream> // std::cout
#include <iomanip>  // std::setw

namespace Helper
{
  /**************************************************************************/
  /*!
    \brief
      Reset the board when a boat which is being placed down overlaps another
      boat
    
    \param [in, out] ocean
      Ocean that contains the 2D grid map layout and to have it's data reset
    
    \param [in] count
      How much was the boat being placed
    
    \param [in] boat
      The boat that is currently being placed, and to have it's data erased
      from the map
  */
  /**************************************************************************/
  void ResetBoard(HLP2::WarBoats::Ocean& ocean, const int& count, const HLP2::WarBoats::Boat& boat)
  {
      int mul_x = 0, mul_y = 0;
      if (HLP2::WarBoats::Orientation::oHORIZONTAL == boat.orientation)
        mul_x = 1;
      else if (HLP2::WarBoats::Orientation::oVERTICAL == boat.orientation)
        mul_y = 1;

      for(int i = 0; i < count; ++i)
      {
        int index = (boat.position.y + (i * mul_y)) * ocean.x_size + (boat.position.x + (i * mul_x));
        *(ocean.grid + index) = 0;
      } 
  }
}

namespace HLP2 
{
  namespace WarBoats 
  {
    int const BOAT_LENGTH { 4 };   //!< Length of a boat
    int const HIT_OFFSET  { 100 }; //!< Add this to the boat ID

    /**************************************************************************/
    /*!
      \brief
        Creates and return a pointer to a dynamically allocated Ocean with
        2D array and an array of boats
      
      \param [in] num_boats
        Total number of boats to be placed
      
      \param [in] x_size
        Horizontal size of the 2D array (Column)
      
      \param [in] y_size
        Vertical size of the 2D array (Row)

      \return
        Pointer to a dynamicalled allocated Ocean with default values
    */
    /**************************************************************************/
    Ocean* CreateOcean(int num_boats, int x_size, int y_size)
    {
      // Ocean
      Ocean* ocean = new Ocean;
      
      // Grid
      ocean->grid = new int[x_size * y_size];
      for(int i = 0; i < x_size; ++i)
      {
        for(int j = 0; j < y_size; ++j)
          *(ocean->grid + i * y_size + j) = DamageType::dtOK;
      }
      ocean->x_size = x_size;
      ocean->y_size = y_size;

      // Boats
      ocean->boats = new Boat[num_boats];
      for(int i = 0; i < num_boats; ++i)
      {
        (ocean->boats + i)->hits = 0;
        (ocean->boats + i)->ID = 0;
      }
      ocean->num_boats = num_boats;

      // ShotStats
      ocean->stats.duplicates = 0;
      ocean->stats.hits = 0;
      ocean->stats.misses = 0;
      ocean->stats.sunk = 0;

      return ocean;
    }

    /**************************************************************************/
    /*!
      \brief
        Deallocate memory on the heap
      
      \param [in] theOcean
        Memory to be dynamically deallocated
    */
    /**************************************************************************/
    void DestroyOcean(Ocean *theOcean)
    {
      delete[] theOcean->boats;
      delete[] theOcean->grid;
      delete theOcean;
    }

    /**************************************************************************/
    /*!
      \brief
        Based on the position of the attack, update ocean accordingly
      
      \param [in, out] ocean
        Ocean that stores data on the position of everything on the 2D array
      
      \param [in] coordinate
        X and Y coordinate of the attack

      \return
        The result of the shots:
        srILLEGAL   - If the shot is outside of the grid
        srMISS      - Hits an open water area
        srDUPLICATE - Hits the same open water area or boat more than once
        srSUNK      - When the boat has successfully sunk
        srHIT       - When the shot hits part of the boat
    */
    /**************************************************************************/
    ShotResult TakeShot(Ocean& ocean, Point const& coordinate)
    {
      if (0 > coordinate.x || ocean.x_size <= coordinate.x || 0 > coordinate.y || ocean.y_size <= coordinate.y)
        return ShotResult::srILLEGAL;

      const int INDEX { coordinate.y * ocean.x_size + coordinate.x };
      const int VALUE { *(ocean.grid + INDEX) };

      // Open water
      if (DamageType::dtOK == VALUE)
      {
        ++ocean.stats.misses;
        *(ocean.grid + INDEX) = DamageType::dtBLOWNUP;
        return ShotResult::srMISS;
      }
      // Rehitting open water                 || shooting at damaged ships
      else if (DamageType::dtBLOWNUP == VALUE || VALUE > HIT_OFFSET)
      {
        ++ocean.stats.duplicates;
        return ShotResult::srDUPLICATE;
      }

      // Hits ship
      ++ocean.stats.hits;
      *(ocean.grid + INDEX) += HIT_OFFSET;
      ++(ocean.boats + VALUE - 1)->hits;
      
      if(BOAT_LENGTH <= (ocean.boats + VALUE - 1)->hits)
      {
        ++ocean.stats.sunk;
        return ShotResult::srSUNK;
      }
      
      return ShotResult::srHIT;
    }

    /**************************************************************************/
    /*!
      \brief
        Place boat in an open water area and store the result in ocean
      
      \param [in, out] ocean
        Container that stores the data of the current map
      
      \param [in] boat
        Boat containing information of the position and orientation
        
      \return
        bpREJECTED - If initial position is out of bound 
                     If initial position is overlapping placed boat
                     If boat goes out of bound vertically or horizontally
                     If while placing a boat, it overlaps another boat
        bpACCEPTED - The boat is placed down onto the grid successfully
    */
    /**************************************************************************/
    BoatPlacement PlaceBoat(Ocean& ocean, Boat const& boat)
    {
      // Current position out of grid
      if (0 > boat.position.x || ocean.x_size <= boat.position.x || 0 > boat.position.y || ocean.y_size <= boat.position.y)
        return BoatPlacement::bpREJECTED;
      // Current position is not overlapping placed boats
      if (0 != *(ocean.grid + boat.position.y * ocean.x_size + boat.position.x))
        return BoatPlacement::bpREJECTED;
      
      // Placing boat onto ocean's grid
      int mul_x = 0, mul_y = 0;
      if (Orientation::oHORIZONTAL == boat.orientation)
        mul_x = 1;
      else if (Orientation::oVERTICAL == boat.orientation)
        mul_y = 1;

      int count { 0 };
      const int VERTICAL_CONDITION { ocean.x_size * ocean.y_size }, HORIZONTAL_CONDITION { ocean.x_size * (boat.position.y + 1 + (mul_y * BOAT_LENGTH)) };
      
      for (int i = 0; i < BOAT_LENGTH; ++i)
      {
        int index = (boat.position.y + (i * mul_y)) * ocean.x_size + (boat.position.x + (i * mul_x));
        // Check if exceeds vertically  || Check if exceeds horizontal   || Check for overlapping
        if (VERTICAL_CONDITION <= index || HORIZONTAL_CONDITION <= index || DamageType::dtOK != *(ocean.grid + index))
        {
          // Reset data on grid to become 0
          Helper::ResetBoard(ocean, count, boat);
          return BoatPlacement::bpREJECTED;
        }        

        *(ocean.grid + index) = boat.ID;
        ++count;
      }
      
      return BoatPlacement::bpACCEPTED;
    }

    /**************************************************************************/
    /*!
      \brief
        Get current shots status happening in the ocean
      
      \param [in] ocean
        Ocean that contains the shots status to what is happening
        
      \return
        Shots status
    */
    /**************************************************************************/
    ShotStats GetShotStats(Ocean const& ocean)
    {
      return ocean.stats;
    }

    /**************************************************************************/
    /*!
      \brief
        Prints the grid (ocean) to the screen.
      
      \param ocean
        The Ocean to print
      
      \param field_width
        How much space each position takes when printed.
      
      \param extraline
        If true, an extra line is printed after each row. If false, no extra
        line is printed.
        
      \param showboats
        If true, the boats are shown in the output. (Debugging feature)
    */
    /**************************************************************************/
    void DumpOcean(const HLP2::WarBoats::Ocean &ocean, int field_width, bool extraline, bool showboats) 
    {
      for (int y = 0; y < ocean.y_size; y++) 
      { // For each row
        for (int x = 0; x < ocean.x_size; x++) 
        { // For each column
            // Get value at x/y position
          int value = ocean.grid[y * ocean.x_size + x];
            // Is it a boat that we need to keep hidden?
          value = ( (value > 0) && (value < HIT_OFFSET) && (showboats == false) ) ? 0 : value;
          std::cout << std::setw(field_width) << value;
        }
        std::cout << "\n";
        if (extraline) { std::cout << "\n"; }
      }
    }
  } // namespace WarBoats
} // namespace HLP2
