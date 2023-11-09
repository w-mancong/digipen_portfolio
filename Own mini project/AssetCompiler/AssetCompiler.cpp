/********************************************************************************************************************************************************
#    All content © 2023 DigiPen Institute of Technology Singapore, all rights reserved.
#    Academic Year:			Trimester 1 Fall 2023
#    Team Name:				Horroscope
#    Game Name:				The Vessel
#    Module:				CSD3400
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#    File Name:				AssetCompiler.cpp
#    Primary Author:		Joachim
#    Secondary Author:		-
*********************************************************************************************************************************************************/

// --------------------
// Header Includes
// --------------------
#include <iostream>

#include "AssetCompiler.h"


// ----------------------------------------
// Compile Asset
// ----------------------------------------
bool AssetCompiler::Compile(const std::string& _inputFilepath) {
	std::cout << "> Preparing asset for compilation...\n> Asset file name: " << _inputFilepath << "\n> Identifying asset type...\n";
	
	// Identify asset type and call respective compilers
	size_t extensionIndex{ _inputFilepath.find_last_of(".") };
	std::string fileExtension{ _inputFilepath.substr(extensionIndex, _inputFilepath.size() - extensionIndex) };

	if (fileExtension == ".fbx" || fileExtension == ".obj")
	{
		std::cout << "> Identified as model/mesh\n";
		if (!geomCompiler.Compile(_inputFilepath))
			return false;
	}
	else 
	{
		std::cout << "> Unable to identify asset type from known types\n";
		return false;
	}

	std::cout << "> Compilation of " << _inputFilepath << " complete.\n";
	return true;

	//// Image/texture
	//if (fileExtension == ".png" ||
	//	fileExtension == ".jpg" ||
	//	fileExtension == ".bmp" ||
	//	fileExtension == ".tga") {
	//	std::cout << "> Identified as texture\n";
	//	if (textureCompiler.Compile(_inputFilepath) == false)
	//		return false;
	//}
	//else if (fileExtension == ".fbx") {
	//	std::cout << "> Identified as model/mesh\n";
	//	if (geomCompiler.Compile(_inputFilepath) == false)
	//		return false;
	//}
	//else {
	//	std::cout << "> Unable to identify asset type from known types\n";
	//	return false;
	//}

	//std::cout << "> Compilation of " << _inputFilepath << " complete.\n";
	//return true;
}