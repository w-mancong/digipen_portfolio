/******************************************************************************/
/*!
\file		GameStateMgr.cpp
\author 	Wong Man Cong
\par    	email: w.mancong\@digipen.edu
\date   	07-03-2022
\brief		This file contains implementations to load the correct function
			depending on the current game state

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/

#include "main.h"

// ---------------------------------------------------------------------------
// globals

// variables to keep track the current, previous and next game state
unsigned int	gGameStateInit;
unsigned int	gGameStateCurr;
unsigned int	gGameStatePrev;
unsigned int	gGameStateNext;

// pointer to functions for game state life cycles functions
void (*GameStateLoad)()		= 0;
void (*GameStateInit)()		= 0;
void (*GameStateUpdate)()	= 0;
void (*GameStateDraw)()		= 0;
void (*GameStateFree)()		= 0;
void (*GameStateUnload)()	= 0;

/*!**************************************************************************************
\brief
	Initialize the game state manager with a starting game state

\param [in] gameStateInit
	Initial game state of the game when application just started
****************************************************************************************/
void GameStateMgrInit(unsigned int gameStateInit)
{
	// set the initial game state
	gGameStateInit = gameStateInit;

	// reset the current, previoud and next game
	gGameStateCurr = 
	gGameStatePrev = 
	gGameStateNext = gGameStateInit;

	// call the update to set the function pointers
	GameStateMgrUpdate();
}

/*!**************************************************************************************
\brief
	Updates the function pointers to the relevant functions whenever there is a change
	of game states
****************************************************************************************/
void GameStateMgrUpdate()
{
	if ((gGameStateCurr == GS_RESTART) || (gGameStateCurr == GS_QUIT))
		return;

	switch (gGameStateCurr)
	{
		case GS_MENU:
		{
			GameStateLoad	= MenuStateLoad;
			GameStateInit	= MenuStateInit;
			GameStateUpdate = MenuStateUpdate;
			GameStateDraw	= MenuStateDraw;
			GameStateFree	= MenuStateFree;
			GameStateUnload = MenuStateUnload;
			break;
		}
		case GS_PLATFORM_1:
		case GS_PLATFORM_2:
		{
			GameStateLoad	= GameStatePlatformLoad;
			GameStateInit	= GameStatePlatformInit;
			GameStateUpdate = GameStatePlatformUpdate;
			GameStateDraw	= GameStatePlatformDraw;
			GameStateFree	= GameStatePlatformFree;
			GameStateUnload = GameStatePlatformUnload;
			break;
		}
		default:
			AE_FATAL_ERROR("invalid state!!");
	}
}
