/********************************************************************************************************************************************************
#    All content © 2023 DigiPen Institute of Technology Singapore, all rights reserved.
#    Academic Year:			Trimester 1 Fall 2023
#    Team Name:				Horroscope
#    Game Name:				The Vessel
#    Module:				CSD3400
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#    File Name:				main.cpp
#    Primary Author:		Joachim
#    Secondary Author:		-
*********************************************************************************************************************************************************/

// --------------------
// Header Includes
// --------------------
#include <iostream>
#include <filesystem>
#include <chrono>
#include <thread>

#include "AssetCompiler.h"

// --------------------
// Constant Defines
// --------------------
const std::string AssetFilePath{ "Assets" };

// ----------------------------------------
// Main Program Logic
// ----------------------------------------
#if NDEBUG
	#pragma comment(linker, "/SUBSYSTEM:windows /ENTRY:mainCRTStartup")
#endif
int main(int argc, char* argv[]) {
	std::cout << "=========== Welcome to Asset Compiler ===========\n";
	std::cout << "Usage: ./AssetCompiler.exe <Asset 1 filepath> <Asset 2 filepath> ...\n";
	std::cout << "Arguments are optional, if none are given, program will run through all assets in its defined assets folder\n";
	std::cout << "File path given is assumed to be absolute filepath. If not found, the program will attempt to use relative file path\n";
	std::this_thread::sleep_for(std::chrono::milliseconds(250));
	
	std::cout << "---------- Asset Compiler Program Start ----------\n";
	AssetCompiler compiler{};
	// No input file given
	if (argc <= 1) {
		for (const auto& asset : std::filesystem::recursive_directory_iterator(AssetFilePath)) {
			std::cout << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n";
			std::cout << "Checking " << asset.path() << " ...\n";
			if (asset.is_regular_file())
				compiler.Compile(asset.path().string());
			else
				std::cout << "Not a regular file. Skipping\n";
			std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
		}
	}
	else {
		for (int i{ 1 }; i < argc; ++i) {
			std::cout << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n";
			compiler.Compile(argv[i]);
			std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
		}
	}

	std::cout << "---------- Asset Compiler Program End ----------\n";
#if _DEBUG
	std::system("pause");
#endif
}
