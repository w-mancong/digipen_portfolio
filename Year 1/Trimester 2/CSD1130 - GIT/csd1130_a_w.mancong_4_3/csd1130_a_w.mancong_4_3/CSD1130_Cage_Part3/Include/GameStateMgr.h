/*!*****************************************************************************
\file		GameStateMgr.h
\author 	Wong Man Cong
\par    	DP email: w.mancong\@digipen.edu
\date   	04-04-2022
\brief
This file contain the appropriate function pointers and variables to manage
the flow of game states

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*******************************************************************************/

#ifndef CSD1130_GAME_STATE_MGR_H_
#define CSD1130_GAME_STATE_MGR_H_

// ---------------------------------------------------------------------------

#include "AEEngine.h"

// ---------------------------------------------------------------------------
// include the list of game states

#include "GameStateList.h"

// ---------------------------------------------------------------------------
// externs

extern GS_STATE gGameStateInit;
extern GS_STATE gGameStateCurr;
extern GS_STATE gGameStatePrev;
extern GS_STATE gGameStateNext;

// ---------------------------------------------------------------------------

extern void (*GameStateLoad)();
extern void (*GameStateInit)();
extern void (*GameStateUpdate)();
extern void (*GameStateDraw)();
extern void (*GameStateFree)();
extern void (*GameStateUnload)();

// ---------------------------------------------------------------------------
// Function prototypes

/*!*****************************************************************************
\brief
	Initializes the game state manager
\param [in] gameStateInit:
	Initial game state of the simulation
*******************************************************************************/
void GameStateMgrInit(GS_STATE gameStateInit);

/*!*****************************************************************************
\brief
	Updates the game state manager when there is a change in game state
*******************************************************************************/
void GameStateMgrUpdate();

// ---------------------------------------------------------------------------

#endif // CSD1130_GAME_STATE_MGR_H_