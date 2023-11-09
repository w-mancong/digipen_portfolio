/********************************************************************************************************************************************************
#    All content © 2023 DigiPen Institute of Technology Singapore, all rights reserved.
#    Academic Year:			Trimester 1 Fall 2023
#    Team Name:				Horroscope
#    Game Name:				The Vessel
#    Module:				CSD3400
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#    File Name:				TexCompiler.cpp
#    Primary Author:		Joachim
#    Secondary Author:		-
*********************************************************************************************************************************************************/

// --------------------
// Header Includes
// --------------------
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <Windows.h>

#include "TexCompiler.h"
#include "ShellApi.h"

// ----------------------------------------
// Compile Textures
// ----------------------------------------
bool TexCompiler::Compile(const std::string& _inputFilepath) {

	std::ifstream fileStream(_inputFilepath);
	if (fileStream.good()) {
		std::cout << "File exists and can be opened." << std::endl;
		// You can read or process the file here
		fileStream.close(); // Close the file when done
	}
	else {
		std::cerr << "File does not exist or cannot be opened." << std::endl;



	}
	std::cout << ">> Loading texture data from " << _inputFilepath << " ...\n";

	// Create directory if doesnt exist
	//	DirectXTex exe throws error if the directory cannot be found
	std::filesystem::create_directory(this->m_OutputFileDirectory);

	std::string executeCommand{ _inputFilepath + this->m_Flags + "-o " + this->m_OutputFileDirectory };

	//std::cout << "Execute command line: " << executeCommand << std::endl;

	HINSTANCE result{ ShellExecuteA(NULL, "open", this->m_DirectXTexExeFilePath.c_str(), executeCommand.c_str(), "x64\\Release\\", SW_HIDE)};

	if ((int)result <= 32)
		return false;

	std::cout << ">> \n";
	return true;

}
