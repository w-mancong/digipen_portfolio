/*!****************************************************************************************************
\file GameStateList.h
\author Wong Man Cong, w.mancong, 2100685
\par DP email: w.mancong\@digipen.edu
\par Course: Game Implementation Technique
\par Class: A
\par Assignment 2
\date 23-01-2022
\brief
	Contains enum for different stages

		Copyright (C) 2022 DigiPen Institute of Technology.
		Reproduction or disclosure of this file or its contents
		without the prior written consent of DigiPen Institute of Technology is prohibited.
******************************************************************************************************/
#pragma once

enum GS_STATES
{
	GS_LEVEL1 = 0,
	GS_LEVEL2,

	GS_QUIT,
	GS_RESTART
};