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
#include "CompiledMesh.h"

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

	void ProcessNode(aiNode* node, MC::CompiledMesh& data, const aiMatrix4x4& parentTransform = aiMatrix4x4()) const;
	void LoadMaterial(MC::Submesh& submesh, aiMesh const* currMesh) const;
	void LoadVertices(MC::Submesh& submesh, aiMesh const* currMesh) const;
	void LoadIndices(MC::Submesh& submesh, aiMesh const* currMesh) const;
	void OptimizeMesh(MC::Submesh& submesh) const;
	void ExtractBoneWeightForVertices(std::vector<Vertex>& vertices, std::vector<MC::BoneProps>& boneProps, aiMesh const* const mesh) const;

	bool Deserialize(std::string const& outputFile, MC::CompiledMesh const& data);
	void Serialize(std::string const& inputFile);
};

#endif
