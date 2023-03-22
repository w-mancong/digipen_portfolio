/******************************************************************************/
/*!
\file		GameStateList.h
\author 	Wong Man Cong
\par    	email: w.mancong\@digipen.edu
\date   	07-03-2022
\brief		This file contains all the enumeration for my game states

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/

#ifndef CSD1130_GAME_STATE_LIST_H_
#define CSD1130_GAME_STATE_LIST_H_

// ---------------------------------------------------------------------------
// game state list

enum
{
	// list of all game states 
	GS_MENU = 0,
	GS_PLATFORM_1,
	GS_PLATFORM_2,
	
	// special game state. Do not change
	GS_RESTART,
	GS_QUIT, 
	GS_NUM
};

// ---------------------------------------------------------------------------

#endif // CSD1130_GAME_STATE_LIST_H_