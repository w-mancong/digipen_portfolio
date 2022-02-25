/*!*****************************************************************************
\file ocean.h
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
////////////////////////////////////////////////////////////////////////////////
#ifndef OCEAN_H
#define OCEAN_H
////////////////////////////////////////////////////////////////////////////////

namespace HLP2 
{
  namespace WarBoats 
  {
    int const BOAT_LENGTH { 4 };   //!< Length of a boat
    int const HIT_OFFSET  { 100 }; //!< Add this to the boat ID

    struct Ocean; //!< Forward declaration for the Ocean 

    enum Orientation   { oHORIZONTAL, oVERTICAL };
    enum ShotResult    { srHIT, srMISS, srDUPLICATE, srSUNK, srILLEGAL };
    enum DamageType    { dtOK = 0, dtBLOWNUP = -1 };
    enum BoatPlacement { bpACCEPTED, bpREJECTED };

      //! A coordinate in the Ocean
    struct Point 
    {
      int x; //!< x-coordinate (column)
      int y; //!< y-coordinate (row)
    };

      //! A boat in the Ocean
    struct Boat 
    {
      int hits;                 //!< Hits taken so far
      int ID;                   //!< Unique ID 
      Orientation orientation;  //!< Horizontal/Vertical
      Point position;           //!< x-y coordinate (left-top)
    };

      //! Statistics of the "game"
    struct ShotStats 
    {
      int hits;       //!< The number of boat hits
      int misses;     //!< The number of boat misses
      int duplicates; //!< The number of duplicate (misses/hits)
      int sunk;       //!< The number of boats sunk
    };
  } // namespace WarBoats

} // namespace HLP2

namespace HLP2 
{
  namespace WarBoats 
  {
      //! The attributes of the ocean
    class Ocean 
    {
      public:
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
        Ocean(int num_boats, int x_size, int y_size);

        /**************************************************************************
          \brief
            Destructor: deallocate memory allocated for boats and grid
        **************************************************************************/
        ~Ocean(void);
        
        /**************************************************************************
          \brief
            Accessor to int* grid
          
          \return 
            Pointer to grid      
        **************************************************************************/
        int* GetGrid(void) const;

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
        ShotResult TakeShot(Point const& coordinate);

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
        BoatPlacement PlaceBoat(Boat const& boat) const;
        
        /**************************************************************************
          \brief
            Return the grid size

          \return
            A point storing column and width of grid
        **************************************************************************/
        Point GetDimensions(void) const;

        /**************************************************************************
          \brief
            Get current shots status happening in the ocean
            
          \return
            Shots status
        **************************************************************************/
        ShotStats GetShotStats(void) const; 
        
      private:      
        int *grid;        //!< The 2D ocean 
        Boat *boats;      //!< The dynamic array of boats
        int num_boats;    //!< Number of boats in the ocean
        int x_size;       //!< Ocean size along x-axis
        int y_size;       //!< Ocean size along y-axis
        ShotStats stats;  //!< Status of the attack
    };
  }
}

#endif // OCEAN_H
////////////////////////////////////////////////////////////////////////////////
