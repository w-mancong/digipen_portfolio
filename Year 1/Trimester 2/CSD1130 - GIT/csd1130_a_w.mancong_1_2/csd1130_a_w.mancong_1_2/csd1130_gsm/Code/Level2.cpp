/*!****************************************************************************************************
\file Level2.cpp
\author Wong Man Cong, w.mancong, 2100685
\par DP email: w.mancong\@digipen.edu
\par Course: Game Implementation Technique
\par Class: A
\par Assignment 2
\date 23-01-2022
\brief
	Contains all the relevant functions for Level 2

		Copyright (C) 2022 DigiPen Institute of Technology.
		Reproduction or disclosure of this file or its contents
		without the prior written consent of DigiPen Institute of Technology is prohibited.
******************************************************************************************************/
#include "pch.h"
#include "Level2.h"

int Level2_Counter{ 0 }, Level2_Lives{ 0 };

/**************************************************************************/
/*!
  \brief
	Loads in necessary assets for level 2 and load in a default value for
	Level2_Lives
*/
/**************************************************************************/
void Level2_Load(void)
{
	ofs << "Level2:Load" << std::endl;
	std::ifstream ifs("Data/Level2_Lives.txt");
	ifs >> Level2_Lives;
	ifs.close();
}

/**************************************************************************/
/*!
  \brief
	Initialize all the necessary data for level 2
*/
/**************************************************************************/
void Level2_Initialize(void)
{
	ofs << "Level2:Initialize" << std::endl;
	std::ifstream ifs("Data/Level2_Counter.txt");
	ifs >> Level2_Counter;
	ifs.close();
}

/**************************************************************************/
/*!
  \brief
	When counter goes to 0, if Level2_Lives > 0, level will restart
	else we will quit the game
*/
/**************************************************************************/
void Level2_Update(void)
{
	ofs << "Level2:Update" << std::endl;
	--Level2_Counter;
	if (0 >= Level2_Counter)
	{
		--Level2_Lives;
		if (0 >= Level2_Lives)
			next = GS_STATES::GS_QUIT;
		else if (0 < Level2_Lives)
			next = GS_STATES::GS_RESTART;
	}
}

/**************************************************************************/
/*!
  \brief
	Draws stuff relevant to level 2
*/
/**************************************************************************/
void Level2_Draw(void)
{
	ofs << "Level2:Draw" << std::endl;
}

/**************************************************************************/
/*!
  \brief
	Deallocate any memory that was allocated for level 2
*/
/**************************************************************************/
void Level2_Free(void)
{
	ofs << "Level2:Free" << std::endl;
}

/**************************************************************************/
/*!
  \brief
	Unload all assets for level 2
*/
/**************************************************************************/
void Level2_Unload(void)
{
	ofs << "Level2:Unload" << std::endl;
}