/********************************************************************************************************************************************************
#    All content © 2023 DigiPen Institute of Technology Singapore, all rights reserved.
#    Academic Year:			Trimester 1 Fall 2023
#    Team Name:				Horroscope
#    Game Name:				The Vessel
#    Module:				CSD3400
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#    File Name:				AssetCompiler.h
#    Primary Author:		Joachim
#    Secondary Author:		-
*********************************************************************************************************************************************************/
#ifndef _ASSET_COMPILER_H
#define _ASSET_COMPILER_H

// --------------------
// Header Includes
// --------------------
#include "GeomCompiler.h"
#include "TexCompiler.h"

// ----------------------------------------
// Asset Compiler Class
// ----------------------------------------
class AssetCompiler {
private:
	GeomCompiler geomCompiler;
	TexCompiler textureCompiler;
public:
	// ----------------------------------------
	// Compile Asset
	// ----------------------------------------
	bool Compile(const std::string& _inputFilePath);
};

#endif