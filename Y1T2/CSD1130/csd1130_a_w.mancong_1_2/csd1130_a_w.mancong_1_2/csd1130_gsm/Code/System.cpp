/*!****************************************************************************************************
\file System.cpp
\author Wong Man Cong, w.mancong, 2100685
\par DP email: w.mancong\@digipen.edu
\par Course: Game Implementation Technique
\par Class: A
\par Assignment 2
\date 23-01-2022
\brief
	Contains function to initialize and terminate our system

		Copyright (C) 2022 DigiPen Institute of Technology.
		Reproduction or disclosure of this file or its contents
		without the prior written consent of DigiPen Institute of Technology is prohibited.
******************************************************************************************************/
#include "System.h"
#include "pch.h"

extern std::ofstream ofs;

/**************************************************************************/
/*!
  \brief
	Initialize ofs
*/
/**************************************************************************/
void System_Initalize(void)
{
	ofs.open("Data/Output.txt");
	ofs << "System:Initialize" << std::endl;
}

/**************************************************************************/
/*!
  \brief
	Close ofs
*/
/**************************************************************************/
void System_Terminate(void)
{
	ofs << "System:Exit" << std::endl;
	ofs.close();
}