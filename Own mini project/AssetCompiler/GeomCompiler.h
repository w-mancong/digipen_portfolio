/********************************************************************************************************************************************************
#    All content � 2023 DigiPen Institute of Technology Singapore, all rights reserved.
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
#include "Animation.h"

// ----------------------------------------
// Geom Compiler Class
// ----------------------------------------
namespace MC = MeshCompiler;
namespace Ani = Animation;
class GeomCompiler {
private:
	const aiScene* m_Scene{ nullptr };
	aiMatrix4x4			m_DescriptorMatrix{};
	Assimp::Importer	m_Importer{};
	Assimp::Exporter	m_Exporter{};
	std::vector<uint32_t> m_AssimpNodeDataChildren{};
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

	// Mesh extraction
	void ProcessNode(aiNode* node, MC::CompiledMesh& data, const aiMatrix4x4& parentTransform = aiMatrix4x4()) const;
	void LoadMaterial(MC::Submesh& submesh, aiMesh const* currMesh) const;
	void LoadVertices(MC::Submesh& submesh, aiMesh const* currMesh) const;
	void LoadIndices(MC::Submesh& submesh, aiMesh const* currMesh) const;
	void OptimizeMesh(MC::Submesh& submesh) const;
	bool Deserialize(std::string const& outputFile, MC::CompiledMesh const& data);
	void Serialize(std::string const& inputFile);

	// Animation 
	void ExtractBoneWeightForVertices(std::vector<Vertex>& vertices, std::vector<Animation::BoneProps>& boneProps, aiMesh const* const mesh) const;
	bool ProcessAnimation(MC::CompiledMesh& data, MC::AnimationData& aniData) const;
	void GenerateBoneTree(aiNode const* src, MC::BoneTreeData& boneData) const;
	void LoadIntermediateBones(aiAnimation const* animation, std::vector<Animation::BoneProps>& boneProps, MC::AnimationData& aniData) const;
	void DeserializeAnimation(std::string const& outputFile, MC::AnimationData const& aniData) const;
	void SerializeAnimation(std::string const& inputFile) const;
	void SerializeBoneTree(Ani::AssimpNodeData* parent, MC::BoneTreeData const& data, uint32_t& idx) const;
};

#endif
