/******************************************************************************/
/*!
\file		Menu.cpp
\author 	Wong Man Cong
\par    	email: w.mancong\@digipen.edu
\date   	07-03-2022
\brief		This file contains implementation for a simple main menu

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/
#include "main.h"

/*!**************************************************************************************
\brief
	Load function for menu
****************************************************************************************/
void MenuStateLoad(void)
{
	
}

/*!**************************************************************************************
\brief
	Init function for menu
	- Initializes game window background to black
****************************************************************************************/
void MenuStateInit(void)
{
	AEGfxSetBackgroundColor(0.0f, 0.0f, 0.0f);
}

/*!**************************************************************************************
\brief
	Update function for menu
	- Based on different input from player, changes game state accordingly
****************************************************************************************/
void MenuStateUpdate(void)
{
	if		(AEInputCheckTriggered(AEVK_1))
		gGameStateNext = GS_PLATFORM_1;
	else if (AEInputCheckTriggered(AEVK_2))
		gGameStateNext = GS_PLATFORM_2;
	else if (AEInputCheckTriggered(AEVK_Q))
		gGameStateNext = GS_QUIT;
}

/*!**************************************************************************************
\brief
	Draw function for menu
****************************************************************************************/
void MenuStateDraw(void)
{
	char strBuffer[100];

	const int TOTAL = 3;
	char  str[TOTAL][100]{ "Press '1' for Level 1", "Press '2' for Level 2", "Press 'Q' to Quit" };

	const float x_pos = -0.3f, y_pos = 0.5f;
	AEGfxSetBlendMode(AE_GFX_BM_BLEND);
	sprintf_s(strBuffer, "Platformer");
	AEGfxPrint(fontID, strBuffer, x_pos, y_pos, 1.0f, 1.0f, 0.0f, 0.9f);

	for (int i = 0; i < TOTAL; ++i)
	{
		sprintf_s(strBuffer, *(str + i));
		AEGfxPrint(fontID, strBuffer, x_pos, y_pos - 0.2f * (i + 1), 1.0f, 1.0f, 1.0f, 1.0f);
	}
}

/*!**************************************************************************************
\brief
	Free function for menu
****************************************************************************************/
void MenuStateFree(void)
{

}

/*!**************************************************************************************
\brief
	Unload function for menu
****************************************************************************************/
void MenuStateUnload(void)
{

}