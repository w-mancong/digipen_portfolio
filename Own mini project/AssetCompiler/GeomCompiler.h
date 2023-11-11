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
#include <type_traits>

// ----------------------------------------
// Geom Compiler Class
// ----------------------------------------
class GeomCompiler {
private:	
	const aiScene* m_Scene{ nullptr };
	aiMatrix4x4			m_DescriptorMatrix;
	Assimp::Importer	m_Importer;
	Assimp::Exporter	m_Exporter;
	std::string			m_OutputFileDirectory{ "Assets\\models" },
						m_ExportFileType{ "fbx" };
	static constexpr 
	const unsigned int	m_ImportFlag{ aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_GenNormals },
						m_ExportFlag{  };

	enum class Vec3Attrib
	{
		Position,
		Normals,
		Tangents,
		BiTangents,		
		Total,
	};

	enum class Vec2Attrib
	{
		TextureCoords,
		Total
	};

	static constexpr uint64_t TOTAL_VEC2_ATTRIBUTE = static_cast<uint64_t>(Vec2Attrib::Total);
	static constexpr uint64_t TOTAL_VEC3_ATTRIBUTE = static_cast<uint64_t>(Vec3Attrib::Total);
	static constexpr uint64_t TOTAL_VERTEX_ATTRIBUTE = TOTAL_VEC2_ATTRIBUTE + TOTAL_VEC3_ATTRIBUTE;

	struct Submesh
	{
		std::string				 meshName{};
		std::vector<glm::vec3>   vec3Attrib[TOTAL_VEC3_ATTRIBUTE]{};
		std::vector<glm::vec2>	 vec2Attrib[TOTAL_VEC2_ATTRIBUTE]{};
		std::vector<uint32_t>	 indices{};
		std::vector<std::string> materialPaths{};
	};

	struct DeserializationData 
	{
		std::vector<Submesh> meshInfos{};
	};

	// Using these struct to find the offset of each vertices/indices
	struct HeaderInfo
	{
		uint64_t meshNameSize{};
		uint64_t vec3VerticesCnt[TOTAL_VEC3_ATTRIBUTE]{};
		uint64_t vec2VerticesCnt[TOTAL_VEC2_ATTRIBUTE]{};
		uint64_t indicesCount{};
	};

	void Count(uint64_t v3[], uint64_t v2[], Submesh const& submesh)
	{
		for (uint64_t i{}; i < TOTAL_VEC3_ATTRIBUTE; ++i)
			v3[i] = submesh.vec3Attrib[i].size();
		for (uint64_t i{}; i < TOTAL_VEC2_ATTRIBUTE; ++i)
			v2[i] = submesh.vec2Attrib[i].size();
	}

	template <typename T, typename = std::is_enum<T>>
	uint64_t Idx(T att)
	{
		return static_cast<uint64_t>(att);
	}

public:
	GeomCompiler();

	// ----------------------------------------
	// Compile Mesh/Models
	// ----------------------------------------
	bool Compile(const std::string& _inputFilepath);

	bool Deserialize(std::string const& outputFile, DeserializationData const& data);
	void Serialize(std::string const& inputFile);

	//bool Serialize(const std::string& _filepath, const SerializationData& _data);
	//bool Deserialize(const std::string& _filepath, SerializationData& _data);
	//bool Deserialize(const std::string& _filepath, 
	//					std::vector<glm::vec3>& _vertices, 
	//					std::vector<glm::vec3>& _normals, 
	//					std::vector<glm::vec3>& _tangents, 
	//					std::vector<glm::vec3>& _biTangents, 
	//					std::vector<glm::vec2>& _texCoords, 
	//					std::vector<std::vector<unsigned int>>& _indices);
	//bool BinarySerialize(const std::string& _filepath, const std::vector<Vec3>& _vertices, const std::vector<std::vector<unsigned int>>& _indices);
	//bool BinaryDeserialize(const std::string& _filepath, std::vector<Vec3>& _vertices, std::vector<std::vector<unsigned int>>& _indices);
};

#endif
