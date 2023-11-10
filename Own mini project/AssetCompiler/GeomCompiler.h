/********************************************************************************************************************************************************
#    All content © 2023 DigiPen Institute of Technology Singapore, all rights reserved.
#    Academic Year:			Trimester 1 Fall 2023
#    Team Name:				Horroscope
#    Game Name:				The Vessel
#    Module:				CSD3400
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#    File Name:				GeomCompiler.h
#    Primary Author:		Joachim
#    Secondary Author:		-
*********************************************************************************************************************************************************/
#ifndef _GEOM_COMPILER_H
#define _GEOM_COMPILER_H

// --------------------
// Header Includes
// --------------------
#include "scene.h"
#include "Importer.hpp"
#include "Exporter.hpp"
#include "postprocess.h"
#include "glm/vec3.hpp"
#include "glm/vec2.hpp"

// ----------------------------------------
// Geom Compiler Class
// ----------------------------------------
class GeomCompiler {
private:	
	const aiScene* m_Scene{ nullptr };
	aiMatrix4x4			m_DescriptorMatrix;
	Assimp::Importer	m_Importer;
	Assimp::Exporter	m_Exporter;
	std::string			m_OutputFileDirectory{ "..\\..\\Assets\\models" },
						m_ExportFileType{ "fbx" };
	static constexpr 
	const unsigned int	m_ImportFlag{ aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_GenNormals },
						m_ExportFlag{  };

	struct Submesh
	{
		std::string							meshName{};
		std::vector<glm::vec3>				vertices{}, 
											normals{}, 
											tangents{}, 
											biTangents{};
		std::vector<glm::vec2>				texCoords{};
		std::vector<std::vector<uint32_t>>	indices{};
		std::vector<std::string>			materialPaths{};
	};

	struct SerializationData 
	{
		std::vector<Submesh> meshInfos{};
	};

public:
	GeomCompiler();

	// ----------------------------------------
	// Compile Mesh/Models
	// ----------------------------------------
	bool Compile(const std::string& _inputFilepath);

	bool Serialize(const std::string& _filepath, const SerializationData& _data);
	bool Deserialize(const std::string& _filepath, SerializationData& _data);
	bool Deserialize(const std::string& _filepath, 
						std::vector<glm::vec3>& _vertices, 
						std::vector<glm::vec3>& _normals, 
						std::vector<glm::vec3>& _tangents, 
						std::vector<glm::vec3>& _biTangents, 
						std::vector<glm::vec2>& _texCoords, 
						std::vector<std::vector<unsigned int>>& _indices);
	//bool BinarySerialize(const std::string& _filepath, const std::vector<Vec3>& _vertices, const std::vector<std::vector<unsigned int>>& _indices);
	//bool BinaryDeserialize(const std::string& _filepath, std::vector<Vec3>& _vertices, std::vector<std::vector<unsigned int>>& _indices);
};

#endif