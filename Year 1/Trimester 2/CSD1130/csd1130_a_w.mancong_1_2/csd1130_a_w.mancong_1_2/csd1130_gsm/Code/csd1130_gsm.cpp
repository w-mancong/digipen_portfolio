/*!****************************************************************************************************
\file csd1130_gsm.cpp
\author Wong Man Cong, w.mancong, 2100685
\par DP email: w.mancong\@digipen.edu
\par Course: Game Implementation Technique
\par Class: A
\par Assignment 2
\date 23-01-2022
\brief
    Program main point of entry. Contains flows of the game

        Copyright (C) 2022 DigiPen Institute of Technology.
        Reproduction or disclosure of this file or its contents
        without the prior written consent of DigiPen Institute of Technology is prohibited.
******************************************************************************************************/
#include "pch.h"
#include <iostream>

#include "GameStateManager.h"
#include "System.h"
#include "Input.h"

/**************************************************************************/
/*!
  \brief
    Program point of entry. Contains logic for the flow of the game
*/
/**************************************************************************/
int main()
{
    //Systems initialize
    System_Initalize();

    //GSM initialize
    GSM_Initialize(GS_STATES::GS_LEVEL1);

    while (current != GS_STATES::GS_QUIT)
    {
        if (current != GS_STATES::GS_RESTART)
        {
            GSM_Update();
            fpLoad();
        }
        else
        {
            current = previous;
            next = previous;
        }

        fpInitialize();

        //the game loop
        while(current == next)
        {
            Input_Handle();
            fpUpdate();
            fpDraw();
        }
        fpFree();
        if (next != GS_STATES::GS_RESTART)
            fpUnload();
        previous = current;
        current = next;
    }

    //Systems exit (terminate)
    System_Terminate();

    return 0;
}
