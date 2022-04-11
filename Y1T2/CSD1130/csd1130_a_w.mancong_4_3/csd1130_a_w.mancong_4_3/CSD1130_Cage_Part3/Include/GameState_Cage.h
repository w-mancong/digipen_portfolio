/*!*****************************************************************************
\file		GameState_Cage.h
\author 	Wong Man Cong
\par    	DP email: w.mancong\@digipen.edu
\date   	04-04-2022
\brief
This file contain functions declarations for GameStateCage to
Load, Init, Update, Draw, Free and Unload relavent behaviour for this game state

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*******************************************************************************/

#ifndef CSD1130_GAME_STATE_PLAY_H_
#define CSD1130_GAME_STATE_PLAY_H_
// ---------------------------------------------------------------------------

/*!*****************************************************************************
\brief
	Load function for GameStateCage
*******************************************************************************/
void GameStateCageLoad(void);

/*!*****************************************************************************
\brief
	Init function for GameStateCage
*******************************************************************************/
void GameStateCageInit(void);

/*!*****************************************************************************
\brief
	Update function for GameStateCage
*******************************************************************************/
void GameStateCageUpdate(void);

/*!*****************************************************************************
\brief
	Render function for GameStateCage
*******************************************************************************/
void GameStateCageDraw(void);

/*!*****************************************************************************
\brief
	Set the flag of all game object instance to false and deallocate all
	the memory allocated for sBallData, sWallData and sPillarData
*******************************************************************************/
void GameStateCageFree(void);

/*!*****************************************************************************
\brief
	Releases all the memory allocated on the heap
*******************************************************************************/
void GameStateCageUnload(void);

// ---------------------------------------------------------------------------

#endif // CSD1130_GAME_STATE_PLAY_H_