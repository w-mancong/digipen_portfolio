/********************************************************************************************************************************************************
#    All content © 2023 DigiPen Institute of Technology Singapore, all rights reserved.
#    Academic Year:			Trimester 1 Fall 2023
#    Team Name:				Horroscope
#    Game Name:				The Vessel
#    Module:				CSD3400
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#    File Name:				GeomCompiler.h
#    Primary Author:		Wong Man Cong
#    Secondary Author:		-
*********************************************************************************************************************************************************/
#ifndef _GEOM_COMPILER_H
#define _GEOM_COMPILER_H

// --------------------
// Header Includes
// --------------------
#include <type_traits>
#include "scene.h"
#include "Importer.hpp"
#include "Exporter.hpp"
#include "postprocess.h"
#include "glm/glm.hpp"

struct Vertex
{
	glm::vec3 position{};
	glm::vec3 normal{};
	glm::vec2 uv{};
	glm::vec3 tangent{};

	bool operator==(const Vertex& v) const&
	{
		return position == v.position && normal == v.normal && uv == v.uv;
	}
};

namespace MeshCompiler
{
	aiTextureType constexpr const TEXTURE_TYPE[] = { aiTextureType_DIFFUSE, aiTextureType_AMBIENT, aiTextureType_EMISSIVE,
												 aiTextureType_NORMALS, aiTextureType_SHININESS, aiTextureType_METALNESS };
	uint64_t constexpr const NUM_TEXTURE_TYPE = sizeof(TEXTURE_TYPE) / sizeof(*TEXTURE_TYPE);
	char constexpr const* TEXTURE_NAMES[NUM_TEXTURE_TYPE] = { "albedo: ", "ambient_occulusion: ", "emissive: ",
															  "normal: ", "roughness: ", "metallic: " };

	struct Submesh
	{
		std::string			   meshName{};
		std::vector<Vertex>	   vertices{};
		std::vector<glm::vec3> bitangents{};
		std::vector<uint32_t>  indices{};
		std::string			   materialName{};
		std::string			   materialPaths[NUM_TEXTURE_TYPE]{};
		glm::mat4			   transformMatrix{};
	};

	struct CompiledMesh
	{
		std::vector<Submesh> meshInfos{};
	};

	/*
		Using this struct to find the offset of each submeshes'
		vertices, indices, mesh name and material paths
	*/
	struct HeaderInfo
	{
		uint64_t meshNameSize{};
		uint64_t verticeCount{};
		uint64_t indicesCount{};
		uint64_t materialNameSize{};
		uint64_t materialPathSize[NUM_TEXTURE_TYPE]{};
	};

	template <typename T, typename = std::is_enum<T>>
	uint64_t Idx(T att)
	{
		return static_cast<uint64_t>(att);
	}
}

// ----------------------------------------
// Geom Compiler Class
// ----------------------------------------
namespace MC = MeshCompiler;
class GeomCompiler {
private:
	const aiScene* m_Scene{ nullptr };
	aiMatrix4x4			m_DescriptorMatrix{};
	Assimp::Importer	m_Importer{};
	Assimp::Exporter	m_Exporter{};
	std::string	const	m_OutputFileDirectory{ "..\\Assets\\models" };
	static constexpr
		const unsigned int	m_ImportFlag{ aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_GenNormals },
		m_ExportFlag{  };
public:
	GeomCompiler();

	// ----------------------------------------
	// Compile Mesh/Models
	// ----------------------------------------
	bool Compile(const std::string& _inputFilepath);

	void ProcessNode(aiNode* node, MC::CompiledMesh& data) const;
	void LoadMaterial(MC::Submesh& submesh, aiMesh const* currMesh) const;
	void LoadVertices(MC::Submesh& submesh, aiMesh const* currMesh) const;
	void LoadIndices(MC::Submesh& submesh, aiMesh const* currMesh) const;

	bool Deserialize(std::string const& outputFile, MC::CompiledMesh const& data);
	void Serialize(std::string const& inputFile);
};

#endif
