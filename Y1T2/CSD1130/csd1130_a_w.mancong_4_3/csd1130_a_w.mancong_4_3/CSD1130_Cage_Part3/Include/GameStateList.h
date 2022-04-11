/*!*****************************************************************************
\file		GameStateList.h
\author 	Wong Man Cong
\par    	DP email: w.mancong\@digipen.edu
\date   	04-04-2022
\brief
This file contain all the game state for this collision simulation

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*******************************************************************************/

#ifndef CSD1130_GAME_STATE_LIST_H_
#define CSD1130_GAME_STATE_LIST_H_

// ---------------------------------------------------------------------------
// game state list

enum class GS_STATE
{
	// list of all game states 
	GS_CAGE = 0, 
	
	// special game state. Do not change
	GS_RESTART,
	GS_QUIT, 
	GS_NUM
};

// ---------------------------------------------------------------------------

#endif // CSD1130_GAME_STATE_LIST_H_