/********************************************************************************************************************************************************
#    All content © 2023 DigiPen Institute of Technology Singapore, all rights reserved.
#    Academic Year:			Trimester 1 Fall 2023
#    Team Name:				Horroscope
#    Game Name:				The Vessel
#    Module:				CSD3400
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#    File Name:				TexCompiler.h
#    Primary Author:		Joachim
#    Secondary Author:		-
*********************************************************************************************************************************************************/
#ifndef _TEX_COMPILER_H
#define _TEX_COMPILER_H

// --------------------
// Header Includes
// --------------------

// ----------------------------------------
// Texture Compiler Class
// ----------------------------------------
class TexCompiler {
private:
	const std::string	m_DirectXTexExeFilePath{ ".\\texconv.exe" },
						m_OutputFileDirectory{ "..\\..\\Assets\\textures" },
						m_Flags{ " -r -y " };
public:
	// ----------------------------------------
	// Compile Textures
	// ----------------------------------------
	bool Compile(const std::string& _inputFilepath);
};

#endif