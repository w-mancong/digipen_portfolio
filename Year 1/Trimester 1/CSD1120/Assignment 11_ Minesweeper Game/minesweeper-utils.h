/*!
@file       minesweeper-utils.h
@author     Swavek Wlodkowski (swavek.wlodkowski)
@course     CSD1120F20
@section    AB
@assignment 9
@date       20/11/2020
@brief      This file contains declarations that must be implemented during this
			assignment inside minesweeper-utils.c.
            Do not modify or submit this file.
*//*__________________________________________________________________________*/

#ifndef MINESWEEPER_H
#define MINESWEEPER_H

#include <stdbool.h>

struct Map;
typedef struct Map Map;

#define BOMB 'X'

Map* create_map(
	unsigned short int width,
	unsigned short int height
);

void destroy_map(Map* map);

void set_tile(
	Map* map,
	unsigned short int column,
	unsigned short int row,
	char state,
	bool is_visible
);

void initialize_map(
	Map* map,
	float probability
);

void print_map(const Map* map);

void reveal_all_tiles(Map* map);

bool all_empty_tiles_visible(const Map* map);

bool is_bomb_tile(
	Map* map,
	unsigned short int column,
	unsigned short int row
);

#endif
