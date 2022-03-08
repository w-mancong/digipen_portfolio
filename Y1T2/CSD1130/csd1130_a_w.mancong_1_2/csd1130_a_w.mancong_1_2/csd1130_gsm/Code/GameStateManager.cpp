/*!****************************************************************************************************
\file GameStateManager.cpp
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
#include "pch.h"

#include "GameStateManager.h"
#include "Level1.h"
#include "Level2.h"

int current = 0, previous = 0, next = 0;

FP fpLoad = nullptr, fpInitialize = nullptr, fpUpdate = nullptr, fpDraw = nullptr, fpFree = nullptr, fpUnload = nullptr;

std::ofstream ofs;

/**************************************************************************/
/*!
  \brief
	Give initial values to the states when the program just started

  \param [in] state
	Initial state of the Game State Manager
*/
/**************************************************************************/
void GSM_Initialize(int state)
{
	current = previous = next = state;
	ofs << "GSM:Initialize" << std::endl;
}

/**************************************************************************/
/*!
  \brief
	Update function pointers when changing of levels
*/
/**************************************************************************/
void GSM_Update(void)
{
	//some unfinished code here
	switch (current)
	{
		case GS_STATES::GS_LEVEL1:
		{
			fpLoad		 = Level1_Load;
			fpInitialize = Level1_Initialize;
			fpUpdate	 = Level1_Update;
			fpDraw		 = Level1_Draw;
			fpFree		 = Level1_Free;
			fpUnload	 = Level1_Unload;
			break;
		}
		case GS_STATES::GS_LEVEL2:
		{
			fpLoad		 = Level2_Load;
			fpInitialize = Level2_Initialize;
			fpUpdate	 = Level2_Update;
			fpDraw		 = Level2_Draw;
			fpFree		 = Level2_Free;
			fpUnload	 = Level2_Unload;
			break;
		}
		default:
			break;
	}
	ofs << "GSM:Update" << std::endl;
}