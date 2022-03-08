/*!****************************************************************************************************
\file Level2.h
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
#pragma once
extern std::ofstream ofs;
extern int next;

/**************************************************************************/
/*!
  \brief
	Loads in necessary assets for level 2 and load in a default value for
	Level2_Lives
*/
/**************************************************************************/
void Level2_Load(void);

/**************************************************************************/
/*!
  \brief
	Initialize all the necessary data for level 2
*/
/**************************************************************************/
void Level2_Initialize(void);

/**************************************************************************/
/*!
  \brief
	When counter goes to 0, if Level2_Lives > 0, level will restart
	else we will quit the game
*/
/**************************************************************************/
void Level2_Update(void);

/**************************************************************************/
/*!
  \brief
	Draws stuff relevant to level 2
*/
/**************************************************************************/
void Level2_Draw(void);

/**************************************************************************/
/*!
  \brief
	Deallocate any memory that was allocated for level 2
*/
/**************************************************************************/
void Level2_Free(void);

/**************************************************************************/
/*!
  \brief
	Unload all assets for level 2
*/
/**************************************************************************/
void Level2_Unload(void);