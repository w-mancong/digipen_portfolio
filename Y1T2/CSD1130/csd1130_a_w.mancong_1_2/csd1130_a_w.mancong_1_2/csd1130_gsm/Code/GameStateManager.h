/*!****************************************************************************************************
\file GameStateManager.h
\author Wong Man Cong, w.mancong, 2100685
\par DP email: w.mancong\@digipen.edu
\par Course: Game Implementation Technique
\par Class: A
\par Assignment 2
\date 23-01-2022
\brief
	Contain functions that update and initializes the Game State Manager

		Copyright (C) 2022 DigiPen Institute of Technology.
		Reproduction or disclosure of this file or its contents
		without the prior written consent of DigiPen Institute of Technology is prohibited.
******************************************************************************************************/
#pragma once

typedef void(*FP)(void);

extern int current, previous, next;

extern FP fpLoad, fpInitialize, fpUpdate, fpDraw, fpFree, fpUnload;

/**************************************************************************/
/*!
  \brief
	Give initial values to the states when the program just started

  \param [in] state
	Initial state of the Game State Manager
*/
/**************************************************************************/
void GSM_Initialize(int state);

/**************************************************************************/
/*!
  \brief
	Update function pointers when changing of levels
*/
/**************************************************************************/
void GSM_Update();