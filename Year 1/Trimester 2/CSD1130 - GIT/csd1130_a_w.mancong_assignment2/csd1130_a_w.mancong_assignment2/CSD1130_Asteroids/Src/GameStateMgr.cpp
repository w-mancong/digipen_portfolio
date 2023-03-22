/******************************************************************************/
/*!
\file		GameStateMgr.cpp
\author 	Wong Man Cong
\par    	email: w.mancong\@digipen.edu
\date   	11-02-2022
\brief		Manages the game state

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

/*!**************************************************************************
\brief
	Initializes the initial game state of the program

\param [in] gameStateInit
	Initial game state of the program
***************************************************************************/
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

/*!**************************************************************************
\brief
	Update function pointers to point to the appropriate functions
	when changing of game states
***************************************************************************/
void GameStateMgrUpdate()
{
	if ((gGameStateCurr == GS_RESTART) || (gGameStateCurr == GS_QUIT))
		return;

	switch (gGameStateCurr)
	{
		case GS_ASTEROIDS:
		{
			GameStateLoad	= GameStateAsteroidsLoad;
			GameStateInit	= GameStateAsteroidsInit;
			GameStateUpdate = GameStateAsteroidsUpdate;
			GameStateDraw	= GameStateAsteroidsDraw;
			GameStateFree	= GameStateAsteroidsFree;
			GameStateUnload = GameStateAsteroidsUnload;
			break;
		}	
		default:
			AE_FATAL_ERROR("invalid state!!");
	}
}
