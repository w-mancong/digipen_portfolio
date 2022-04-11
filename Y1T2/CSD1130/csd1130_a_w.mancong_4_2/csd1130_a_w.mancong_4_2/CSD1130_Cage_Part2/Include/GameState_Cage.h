/*!*****************************************************************************
\file		GameState_Cage.h
\author 	Wong Man Cong
\par    	DP email: w.mancong\@digipen.edu
\date   	26-03-2022
\brief
This file contains functions declaration for Load, Init, Update, Draw, Free and Unload
resources that simulates collision between a line segment and a ball

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*******************************************************************************/

#ifndef CSD1130_GAME_STATE_PLAY_H_
#define CSD1130_GAME_STATE_PLAY_H_


// ---------------------------------------------------------------------------

void GameStateCageLoad(void);
void GameStateCageInit(void);
void GameStateCageUpdate(void);
void GameStateCageDraw(void);
void GameStateCageFree(void);
void GameStateCageUnload(void);

// ---------------------------------------------------------------------------

#endif // CSD1130_GAME_STATE_PLAY_H_