/******************************************************************************/
/*!
\file		GameState_Platformer.h
\author 	Wong Man Cong
\par    	email: w.mancong\@digipen.edu
\date   	07-03-2022
\brief		This file contains function declaration for my platformer game

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#ifndef CSD1130_GAME_STATE_PLAY_H_
#define CSD1130_GAME_STATE_PLAY_H_

// ======================================================================================

/*!**************************************************************************************
\brief
	Load function for menu
****************************************************************************************/
void MenuStateLoad(void);

/*!**************************************************************************************
\brief
	Init function for menu
	- Initializes game window background to black
****************************************************************************************/
void MenuStateInit(void);

/*!**************************************************************************************
\brief
	Update function for menu
	- Based on different input from player, changes game state accordingly
****************************************************************************************/
void MenuStateUpdate(void);

/*!**************************************************************************************
\brief
	Draw function for menu
****************************************************************************************/
void MenuStateDraw(void);

/*!**************************************************************************************
\brief
	Free function for menu
****************************************************************************************/
void MenuStateFree(void);

/*!**************************************************************************************
\brief
	Unload function for menu
****************************************************************************************/
void MenuStateUnload(void);

// ======================================================================================

/*!**************************************************************************************
\brief
	Load function of my platformer game
	- Allocates memory on the heap
	- Creating meshes for individual game objects
	- Load in MapData and BinaryCollisionArray from text file
	- Initialize MapTransform matrix
****************************************************************************************/
void GameStatePlatformLoad(void);

/*!**************************************************************************************
\brief
	Init function of my platformer game
	- Initializes all the necessary data to it's default value
****************************************************************************************/
void GameStatePlatformInit(void);

/*!**************************************************************************************
\brief
	Update function of my platformer game
	- Update player's input
	- Spawns particle
	- Update status of particles for when to despawn it
	- Apply gravity to all necessary game objects
	- Update enemy state machine
	- Update position of game object based on their velocity
	- Update bounding box
	- Update binary collision
	- Update aabb collision
	- Computes transformation matrix of each game object
	- Clamp camera's position
****************************************************************************************/
void GameStatePlatformUpdate(void);

/*!**************************************************************************************
\brief
	Draw function for my platformer game
****************************************************************************************/
void GameStatePlatformDraw(void);

/*!**************************************************************************************
\brief
	Reset the active flag of all GameObjInst
****************************************************************************************/
void GameStatePlatformFree(void);

/*!**************************************************************************************
\brief
	Deallocate any memory from the heap
****************************************************************************************/
void GameStatePlatformUnload(void);

// ---------------------------------------------------------------------------
#endif // CSD1130_GAME_STATE_PLAY_H_