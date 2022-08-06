/******************************************************************************/
/*!
\file		GameState_Asteroids.h
\author 	Wong Man Cong
\par    	email: w.mancong\@digipen.edu
\date   	11-02-2022
\brief		Contains functions to load, init, update, draw, free and unload
			asteroid game

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#ifndef CSD1130_GAME_STATE_PLAY_H_
#define CSD1130_GAME_STATE_PLAY_H_

// ---------------------------------------------------------------------------

/*!**************************************************************************
\brief
	Asteroid's load function
***************************************************************************/
void GameStateAsteroidsLoad(void);

/*!**************************************************************************
\brief
	Asteroid's init function
***************************************************************************/
void GameStateAsteroidsInit(void);

/*!**************************************************************************
\brief
	Asteroid's update function
***************************************************************************/
void GameStateAsteroidsUpdate(void);

/*!**************************************************************************
\brief
	Asteroids draw function
***************************************************************************/
void GameStateAsteroidsDraw(void);

/*!**************************************************************************
\brief
	Asteroid's free function
***************************************************************************/
void GameStateAsteroidsFree(void);

/*!**************************************************************************
\brief
	Asteroid's unload function
***************************************************************************/
void GameStateAsteroidsUnload(void);

// ---------------------------------------------------------------------------

#endif // CSD1130_GAME_STATE_PLAY_H_


