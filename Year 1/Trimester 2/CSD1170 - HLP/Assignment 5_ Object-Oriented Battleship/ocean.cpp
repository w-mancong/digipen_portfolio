/*!*****************************************************************************
\file ocean.cpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Assignment 5
\date 13-02-2022
\brief 
A simple Battleship game that can have the size of the board changed according
to how big the playing field the player decides it to be.
*******************************************************************************/
#include "ocean.h"

namespace HLP2 
{
  namespace WarBoats 
  {
    /**************************************************************************
      \brief
        Constructor that allocates memory of grid and array of boats
      
      \param [in] num_boats
        Total number of boats to be placed
      
      \param [in] x_size
        Horizontal size of the 2D array (Column)
      
      \param [in] y_size
        Vertical size of the 2D array (Row)
    **************************************************************************/
    Ocean::Ocean(int num_boats, int x_size, int y_size)
    {
      // Grid
      grid = new int[x_size * y_size];
      for(int i = 0; i < y_size; ++i)
      {
        for(int j = 0; j < x_size; ++j)
          *(grid + i * x_size + j) = DamageType::dtOK;
      }
      this->x_size = x_size;
      this->y_size = y_size;

      // Boats
      boats = new Boat[num_boats];
      for(int i = 0; i < num_boats; ++i)
      {
        (boats + i)->hits = 0;
        (boats + i)->ID = 0;
      }
      this->num_boats = num_boats;

      // ShotStats
      stats = ShotStats{0, 0, 0, 0};
    }

    /**************************************************************************
      \brief
        Destructor: deallocate memory allocated for boats and grid
    **************************************************************************/
    Ocean::~Ocean(void)
    {
      delete[] boats;
      delete[] grid;
    }

    /**************************************************************************
      \brief
        Accessor to int* grid
      
      \return 
        Pointer to grid      
    **************************************************************************/
    int* Ocean::GetGrid(void) const
    {
      return grid;
    }

    /**************************************************************************
      \brief
        Based on the position of the attack, update ocean accordingly
      
      \param [in] coordinate
        X and Y coordinate of the attack

      \return
        The result of the shots:
        srILLEGAL   - If the shot is outside of the grid
        srMISS      - Hits an open water area
        srDUPLICATE - Hits the same open water area or boat more than once
        srSUNK      - When the boat has successfully sunk
        srHIT       - When the shot hits part of the boat
    **************************************************************************/
    ShotResult Ocean::TakeShot(Point const& coordinate)
    {
      if (0 > coordinate.x || x_size <= coordinate.x || 0 > coordinate.y || y_size <= coordinate.y)
        return ShotResult::srILLEGAL;

      const int INDEX { coordinate.y * x_size + coordinate.x };
      const int VALUE { *(grid + INDEX) };

      // Open water
      if (DamageType::dtOK == VALUE)
      {
        ++stats.misses;
        *(grid + INDEX) = DamageType::dtBLOWNUP;
        return ShotResult::srMISS;
      }
      // Rehitting open water                 || shooting at damaged ships
      else if (DamageType::dtBLOWNUP == VALUE || VALUE > HIT_OFFSET)
      {
        ++stats.duplicates;
        return ShotResult::srDUPLICATE;
      }

      // Hits ship
      ++stats.hits;
      *(grid + INDEX) += HIT_OFFSET;
      ++(boats + VALUE - 1)->hits;
      
      if(BOAT_LENGTH <= (boats + VALUE - 1)->hits)
      {
        ++stats.sunk;
        return ShotResult::srSUNK;
      }
      
      return ShotResult::srHIT;
    }

    /**************************************************************************
      \brief
        Place boat in an open water area
        
      \return
        bpREJECTED - If initial position is out of bound 
                     If initial position is overlapping placed boat
                     If boat goes out of bound vertically or horizontally
                     If while placing a boat, it overlaps another boat
        bpACCEPTED - The boat is placed down onto the grid successfully
    **************************************************************************/
    BoatPlacement Ocean::PlaceBoat(Boat const& boat) const
    {
      // Current position out of grid
      if (0 > boat.position.x || x_size <= boat.position.x || 0 > boat.position.y || y_size <= boat.position.y)
        return BoatPlacement::bpREJECTED;
      // Current position is not overlapping placed boats
      if (0 != *(grid + boat.position.y * x_size + boat.position.x))
        return BoatPlacement::bpREJECTED;
      
      // Placing boat onto ocean's grid
      int mul_x = 0, mul_y = 0;
      if (Orientation::oHORIZONTAL == boat.orientation)
      {
        mul_x = 1;
        if(boat.position.x + BOAT_LENGTH > x_size)
          return bpREJECTED;
      }
      else if (Orientation::oVERTICAL == boat.orientation)
      {
        if(boat.position.y + BOAT_LENGTH > y_size)
          return bpREJECTED;
        mul_y = 1;
      }
      
      // Check for overlap
            for (int i = 0; i < BOAT_LENGTH; ++i)
      {
        int index = (boat.position.y + (i * mul_y)) * x_size + (boat.position.x + (i * mul_x));
        // Check if exceeds vertically  || Check if exceeds horizontal   || Check for overlapping
        if (DamageType::dtOK != *(grid + index))
          return bpREJECTED;
      }
      
      for (int i = 0; i < BOAT_LENGTH; ++i)
      {
        int index = (boat.position.y + (i * mul_y)) * x_size + (boat.position.x + (i * mul_x));
        *(grid + index) = boat.ID;
      }
      
      return BoatPlacement::bpACCEPTED;
    }

    /**************************************************************************
      \brief
        Return the grid size

      \return
        A point storing column and width of grid
    **************************************************************************/
    Point Ocean::GetDimensions(void) const
    {
      return Point({ x_size, y_size });
    }

    /**************************************************************************
      \brief
        Get current shots status happening in the ocean
        
      \return
        Shots status
    **************************************************************************/
    ShotStats Ocean::GetShotStats(void) const
    {
      return stats;
    }
  } // namespace WarBoats
} 
