/******************************************************************************/
/*!
\file		GameStateMgr.cpp
\author 	Wong Man Cong
\par    	email: w.mancong\@digipen.edu
\date   	07-03-2022
\brief		This file contains function declarations for game state manager

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#ifndef CSD1130_GAME_STATE_MGR_H_
#define CSD1130_GAME_STATE_MGR_H_

// ======================================================================================

#include "AEEngine.h"

// ======================================================================================
// include the list of game states

#include "GameStateList.h"

// ======================================================================================
// externs

extern unsigned int gGameStateInit;
extern unsigned int gGameStateCurr;
extern unsigned int gGameStatePrev;
extern unsigned int gGameStateNext;

// ======================================================================================

extern void (*GameStateLoad)();
extern void (*GameStateInit)();
extern void (*GameStateUpdate)();
extern void (*GameStateDraw)();
extern void (*GameStateFree)();
extern void (*GameStateUnload)();

// ======================================================================================
// Function prototypes

/*!**************************************************************************************
\brief
	Initialize the game state manager with a starting game state

\param [in] gameStateInit
	Initial game state of the game when application just started
****************************************************************************************/
void GameStateMgrInit(unsigned int gameStateInit);

/*!**************************************************************************************
\brief
	Updates the function pointers to the relevant functions whenever there is a change
	of game states
****************************************************************************************/
void GameStateMgrUpdate();

// ---------------------------------------------------------------------------

#endif // CSD1130_GAME_STATE_MGR_H_