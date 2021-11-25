/*!
@file       minesweeper-utils.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@assignment 11
@date       25/11/2021
@brief      A non-interactive minesweeper game
*//*____________________________________________________________________________________________________________________________*/
#include <stdio.h>
#include <stdlib.h>
#include "minesweeper-utils.h"

typedef struct Tile
{
	char state;
	bool is_visible;
} Tile;

typedef Tile* Row;
typedef struct Map
{
	unsigned short int width;
	unsigned short int height;
	Row* grid;      // Aka Tile**
} Map;

/*!
@brief	create a dynamic 2d map of width and height

@param  width: number of tiles in a column
        height: number of tiles in a row

@return pointer to the base address of map
*//*____________________________________________________________________________________________________________________________*/
Map* create_map(unsigned short int width, unsigned short int height)
{
    Map* map = (Map*)malloc(sizeof(Map));
    map->width = width;
    map->height = height;
    
    // initialising dynamic 2d array  
    size_t len = sizeof(Row) * height + sizeof(Tile) * width * height;
    map->grid = (Row*)malloc(len);
    Row ptr = (Row)(map->grid + height);            // Row aka Tile*
    for(int row = 0; row < height; ++row)
        *(map->grid + row) = (ptr + width * row);

    return map;
}

/*!
@brief	deallocate dynamic memory allocated on the heap

@param  map: reference to map to have it's memory deallocated
*//*____________________________________________________________________________________________________________*/
void destroy_map(Map* map)
{
    if(map)
    {
        if(map->grid)
        {
            free(map->grid);
            map->grid = NULL;
        }        
        free(map);
        map = NULL;
    }
}

/*!
@brief	set the state and visibility of a tile at position row and column

@param  map: reference to the map to set it's tile
        coulmn: position at column
        row: position at row
        state: 'X' if it's a bomb, else a number represent the adjacent bomb
        is_visible: visibility of the tile
*//*____________________________________________________________________________________________________________*/
void set_tile(Map* map, unsigned short int column, unsigned short int row, char state, bool is_visible)
{
    (*(map->grid + row) + column)->is_visible = is_visible;
    (*(map->grid + row) + column)->state = state;
}

/*!
@brief	takes in a map and fill it with values for each tile

@param  map: reference to the map to set a value for each tile
        probability: 
*//*____________________________________________________________________________________________________________*/
void initialize_map(Map* map, float probability)
{
    for(unsigned short row = 0; row < map->height; ++row)
    {
        for(unsigned short col = 0; col < map->width; ++col)
        {
            int random = rand() % 100;
            if(random > (int)(probability * 100))
                set_tile(map, col, row, BOMB, false);
            else
                set_tile(map, col, row, '0', false);
        }
    }

    // check adjacent tiles for bomb
    for(unsigned short row = 0; row < map->height; ++row)
    {
        for(unsigned short col = 0; col < map->width; ++col)
        {
            if((*(map->grid + row) + col)->state == BOMB)
                continue;
            int adjacent_bomb = 0;
            // check adjacent tiles of top row
            if(row - 1 >= 0)            
            {
                if((*(map->grid + (row - 1)) + col)->state == BOMB) // check top
                    ++adjacent_bomb;              
                if(col - 1 >= 0) // check top left
                {
                    if((*(map->grid + (row - 1)) + (col - 1))->state == BOMB)
                        ++adjacent_bomb;
                }
                if(col + 1 < map->width) // check top right
                {
                    if((*(map->grid + (row - 1)) + (col + 1))->state == BOMB)
                        ++adjacent_bomb;
                }
            }

            // check adjacent tiles of the same row
            if(col - 1 >= 0) // check left
            {
                if((*(map->grid + row) + (col - 1))->state == BOMB)
                    ++adjacent_bomb;
            }

            if(col + 1 < map->width) // check right
            {
                if((*(map->grid + row) + (col + 1))->state == BOMB)
                    ++adjacent_bomb;
            }

            // check adjacent tiles of the btm row
            if(row + 1 < map->height)
            {
                if((*(map->grid + (row + 1)) + col)->state == BOMB) // check btm
                    ++adjacent_bomb;
                if(col - 1 >= 0) // check btm left
                {
                    if((*(map->grid + (row + 1)) + (col - 1))->state == BOMB)
                        ++adjacent_bomb;
                }
                if(col + 1 < map->width) // check btm right
                {
                    if((*(map->grid + (row + 1)) + (col + 1))->state == BOMB)
                        ++adjacent_bomb;
                }
            }
            // set current tile state to the number of adjacent bombs
            (*(map->grid + row) + col)->state = (char)((int)'0' + adjacent_bomb);
        }
    }
}

/*!
@brief	Prints the tile one by one, and if its invisible, print a space instead

@param  map: reference to the map
*//*____________________________________________________________________________________________________________*/
void print_map(const Map* map)
{
    printf("  ");
    for(unsigned short row = 0; row < map->width; ++row)
        printf("%d%c", row, row + 1 !=  map->width ? ' ' : '\n');

    for(unsigned short row = 0; row < map->height; ++row)
    {
        printf("%d|", row);
        for(unsigned short col = 0; col < map->width; ++col)
            printf("%c|%s", (*(map->grid + row) + col)->is_visible ? (*(map->grid + row) + col)->state : ' ', col + 1 !=  map->width ? "" : "\n");
    }
}

/*!
@brief	Set all the tiles to in map to be visible

@param  map: reference to the map to have it's tile be visible
*//*____________________________________________________________________________________________________________*/
void reveal_all_tiles(Map* map)
{
    for(unsigned short row = 0; row < map->height; ++row)
    {
        for(unsigned short col = 0; col < map->width; ++col)
            (*(map->grid + row) + col)->is_visible = true;
    }
}

/*!
@brief	Test whether all non-bomb tiles have been revealed

@param  map: reference to the map

@return true if all non-bomb tiles are visible
*//*____________________________________________________________________________________________________________*/
bool all_empty_tiles_visible(const Map* map)
{
    for (unsigned short row = 0; row < map->height; ++row)
    {
        for (unsigned short col = 0; col < map->width; ++col)
        {
            if ((*(map->grid + row) + col)->state != BOMB && !(*(map->grid + row) + col)->is_visible)
                return false;
        }
    }
    return true;
}

/*!
@brief	reveals the tile at column and row. check if the current tile is a bomb

@param  map: reference to the map
        column: position at column
        row: position at row

@return true if the current tile at specified column and row is a bomb
*//*____________________________________________________________________________________________________________*/
bool is_bomb_tile(Map* map, unsigned short int column, unsigned short int row)
{
    if (map->width <= column || map->height <= row)
    {
        printf("Error: wrong column or row specified.\n");
        return false;
    }
    (*(map->grid + row) + column)->is_visible = true;
    if ((*(map->grid + row) + column)->state == BOMB)
        return true;
    return false;
}
