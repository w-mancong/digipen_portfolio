/*!****************************************************************************************************
\file Level1.cpp
\author Wong Man Cong, w.mancong, 2100685
\par DP email: w.mancong\@digipen.edu
\par Course: Game Implementation Technique
\par Class: A
\par Assignment 2
\date 23-01-2022
\brief
	Contains all the relevant functions for Level 1

		Copyright (C) 2022 DigiPen Institute of Technology.
		Reproduction or disclosure of this file or its contents
		without the prior written consent of DigiPen Institute of Technology is prohibited.
******************************************************************************************************/
#include "pch.h"
#include "Level1.h"

int Level1_Counter{ 0 };

/**************************************************************************/
/*!
  \brief
	Loads in necessary assets for level 1
*/
/**************************************************************************/
void Level1_Load(void)
{
	std::ifstream ifs("Data/Level1_Counter.txt");
	ifs >> Level1_Counter;
	ifs.close();
	ofs << "Level1:Load" << std::endl;
}

/**************************************************************************/
/*!
  \brief
	Initialize all the necessary data for level 1
*/
/**************************************************************************/
void Level1_Initialize(void)
{
	ofs << "Level1:Initialize" << std::endl;
}

/**************************************************************************/
/*!
  \brief
	When counter goes to 0, changes to level 2
*/
/**************************************************************************/
void Level1_Update(void)
{
	ofs << "Level1:Update" << std::endl;
	--Level1_Counter;
	if (0 >= Level1_Counter)
		next = GS_STATES::GS_LEVEL2;
}

/**************************************************************************/
/*!
  \brief
	Draws stuff relevant to level 1
*/
/**************************************************************************/
void Level1_Draw(void)
{
	ofs << "Level1:Draw" << std::endl;
}

/**************************************************************************/
/*!
  \brief
	Deallocate any memory that was allocated for level 1
*/
/**************************************************************************/
void Level1_Free(void)
{
	ofs << "Level1:Free" << std::endl;
}

/**************************************************************************/
/*!
  \brief
	Unload all assets for level 1
*/
/**************************************************************************/
void Level1_Unload(void)
{
	ofs << "Level1:Unload" << std::endl;
}