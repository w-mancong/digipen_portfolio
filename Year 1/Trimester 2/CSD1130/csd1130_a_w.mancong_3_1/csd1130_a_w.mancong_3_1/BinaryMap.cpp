/*!**********************************************************************************
\file		BinaryMap.cpp
\author 	Wong Man Cong
\par    	email: w.mancong\@digipen.edu
\date   	February 16, 2022
\brief
This program loads from text file and store into a 2D dynamic array
the Map Data and the Binary Collision data

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior 
written consent of DigiPen Institute of Technology is prohibited.
************************************************************************************/

#include "BinaryMap.h"
#include <iostream>
#include <fstream>
#include <string>

/*The number of horizontal elements*/
static int BINARY_MAP_WIDTH;

/*The number of vertical elements*/
static int BINARY_MAP_HEIGHT;

/*This will contain all the data of the map, which will be retreived from a file
when the "ImportMapDataFromFile" function is called*/
static int **MapData;

/*This will contain the collision data of the binary map. It will be filled in the 
"ImportMapDataFromFile" after filling "MapData". Basically, if an array element 
in MapData is 1, it represents a collision cell, any other value is a non-collision
cell*/
static int **BinaryCollisionArray;

/*!**********************************************************************************
\brief
	Load data from text file and storing it into 2D dynamic array

\param [in] FileName
	Name of the file to have data to be extrated and stored into the 
	MapData and BinaryCollisionArray

\return
	False (0) if FileName does not exist else True (1)
************************************************************************************/
int ImportMapDataFromFile(const char *FileName)
{
	std::ifstream ifs(FileName);
	std::string buffer;

	if(!ifs.is_open())
		return false;

	int count = 0;
	const int DATA = 2;	// just tryna loop to get my width and height value
	int temp[DATA] = {};
	while (count < DATA)
	{
		ifs >> buffer;
		if (std::isdigit(buffer[0]))
			temp[count++] = std::stoi(buffer);
	}

	BINARY_MAP_WIDTH  = temp[0];	// col
	BINARY_MAP_HEIGHT = temp[1];	// row

	// Allocating memory for my 2d array
	MapData = new int* [(size_t)BINARY_MAP_HEIGHT * (size_t)BINARY_MAP_WIDTH];
	int* ptr = (int*)(MapData + BINARY_MAP_HEIGHT);

	for (size_t i = 0; i < BINARY_MAP_HEIGHT; ++i)
		*(MapData + i) = (ptr + (size_t)BINARY_MAP_WIDTH * i);

	BinaryCollisionArray = new int* [(size_t)BINARY_MAP_HEIGHT * (size_t)BINARY_MAP_WIDTH];
	ptr = (int*)(BinaryCollisionArray + BINARY_MAP_HEIGHT);

	for (size_t i = 0; i < BINARY_MAP_HEIGHT; ++i)
		*(BinaryCollisionArray + i) = (ptr + (size_t)BINARY_MAP_WIDTH * i);

	for (size_t i = 0; i < BINARY_MAP_HEIGHT; ++i)
	{
		for (size_t j = 0; j < BINARY_MAP_WIDTH; ++j)
		{
			ifs >> buffer;
			int num = std::stoi(buffer);

			*(*(MapData + i) + j)				= num;
			*(*(BinaryCollisionArray + i) + j)	= num == 1 ? 1 : 0;
		}
	}

	ifs.close();

	return true;
}

/*!**********************************************************************************
\brief
	Deallocate dynamically allocated memory for MapData & BinaryCollisionArray
************************************************************************************/
void FreeMapData(void)
{
	delete[] MapData;
	delete[] BinaryCollisionArray;
}

/*!**********************************************************************************
\brief
	Print out the information stored into MapData
************************************************************************************/
void PrintRetrievedInformation(void)
{
	for (size_t i = 0; i < BINARY_MAP_HEIGHT; ++i)
	{
		for (size_t j = 0; j < BINARY_MAP_WIDTH; ++j)
		{
			std::cout << *(*(MapData + i) + j) << ' ';
		}
		std::cout << std::endl;
	}
}

/*!**********************************************************************************
\brief
	Get the value store inside the position X and Y

\param [in] X
	X coordinate of the array
	
\param [in] Y
	Y coordinate of the array

\return
	Return either 0 or 1 based on the position in the array
************************************************************************************/
int GetCellValue(int X, int Y)
{
	if (0 > X || BINARY_MAP_WIDTH <= X || 0 > Y || BINARY_MAP_HEIGHT <= Y)
		return 0;
	return BinaryCollisionArray[Y][X];
}

/*!**********************************************************************************
\brief
	Snaps the Coordinate back to it's original position
	
\param [in, out] Coordinate
	The coordinate to be snapped back to it's position
************************************************************************************/
void SnapToCell(float *Coordinate)
{
	*Coordinate = static_cast<float>(static_cast<int>(*Coordinate));
	*Coordinate += 0.5f;
}

/*!**********************************************************************************
\brief
	Based on the current position and size of the cube, calculate the hotspots
	of the cube and determine if it has any collision using the datas store inside
	BinaryCollisionArray. If any of the hotspots are inside the cell of the 
	collision array:
	-> The COLLISION_LEFT, COLLISION_RIGHT, COLLISION_TOP, COLLISION_BOTTOM flags
	will be flagged accordingly

\param [in] PosX
	X position of the cube

\param [in] PosY
	Y position of the cube

\param [in] scaleX
	Size of the cube horizontally

\param [in] scaleY
	Size of the cube vertically

\return
	A flag number determining where if there is collision on the TOP, BOTTOM,
	RIGHT and LEFT edges of the cube
************************************************************************************/
int CheckInstanceBinaryMapCollision(float PosX, float PosY, 
									float scaleX, float scaleY)
{
	int flag = 0;
	int x1 = 0, y1 = 0, x2 = 0, y2 = 0;

	// hotspot 1 (right-up)
	x1 = static_cast<int>(PosX + scaleX * 0.5f);
	y1 = static_cast<int>(PosY + scaleY * 0.25f);

	// hotspot 2 (right-down)
	x2 = static_cast<int>(PosX + scaleX * 0.5f);
	y2 = static_cast<int>(PosX - scaleY * 0.25f);
	if (GetCellValue(x1, y1) || GetCellValue(x2, y2))
		flag = flag | COLLISION_RIGHT;

	// hotspot 3 (left-up)
	x1 = static_cast<int>(PosX - scaleX * 0.5f);
	y1 = static_cast<int>(PosY + scaleY * 0.25f);

	// hotspot 4 (left-down)
	x2 = static_cast<int>(PosX - scaleX * 0.5f);
	y2 = static_cast<int>(PosX - scaleY * 0.25f);
	if (GetCellValue(x1, y1) || GetCellValue(x2, y2))
		flag = flag | COLLISION_LEFT;

	// hotspot 5 (up-right)
	x1 = static_cast<int>(PosX + scaleX * 0.25f);
	y1 = static_cast<int>(PosY + scaleY * 0.5f);

	// hotspot 6 (up-leftt)
	x2 = static_cast<int>(PosX - scaleX * 0.25f);
	y2 = static_cast<int>(PosY + scaleY * 0.5f);
	if (GetCellValue(x1, y1) || GetCellValue(x2, y2))
		flag = flag | COLLISION_TOP;

	// hotspot 7 (down-right)
	x1 = static_cast<int>(PosX + scaleX * 0.25f);
	y1 = static_cast<int>(PosY - scaleY * 0.5f);

	// hotspot 8 (down-left)
	x2 = static_cast<int>(PosX - scaleX * 0.25f);
	y2 = static_cast<int>(PosY - scaleY * 0.5f);
	if (GetCellValue(x1, y1) || GetCellValue(x2, y2))
		flag = flag | COLLISION_BOTTOM;

	return flag;
}