#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "glm/glm.hpp"
#include "assimp/material.h"
#include "Vertex.h"
#include "SkinnedModel.h"
#include "Bone.h"
#include "Animation.h"

namespace Ani = Animation;
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
		std::vector<uint32_t>  indices{};
		std::string			   materialName{};
		std::string			   materialPaths[NUM_TEXTURE_TYPE]{};
		glm::mat4			   transformMatrix{};
	};

	struct CompiledMesh
	{
		std::string modelName{};
		std::vector<Submesh> meshInfos{};
		std::vector<Animation::BoneProps> boneProps{};
	};

	struct AnimationData
	{
		float duration{};
		float tps{};
		std::vector<Ani::Bone> bones{};
		Ani::AssimpNodeData rootNode{};
		std::vector<Ani::BoneProps> boneProps{};
		std::string clipName{};
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